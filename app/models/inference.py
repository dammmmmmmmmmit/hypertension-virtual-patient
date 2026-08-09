"""
Inference-time feature construction + prediction, using the models trained
in train.py. This is the low-level prediction layer the Week-2 LangGraph
agent's `run_prediction` node will call — it does NOT parse natural
language (that's the agent's `parse_query`/`resolve_entities` job, not
built yet). Input here is already-resolved drug names.

Scope note: only single drugs and 2-drug combinations are supported,
matching what the model was trained on (see DECISIONS.md #1, #3). 3+ drug
combinations raise NotImplementedError rather than silently extrapolating
past what the training data covers.
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sqlalchemy import select

from app.core.drug_registry import DISCOURAGED_COMBINATIONS, HYPERTENSION_DRUGS
from app.db.models import ResolvedCompound
from app.db.session import async_session_factory
from app.models.dataset import ALL_CLASSES, _rdkit_mean, compute_potency_z_scores
from app.schemas.patient import DiseaseParameterDelta, PatientProfile

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

_NAME_TO_CLASS = {d["name"]: d["drug_class"] for d in HYPERTENSION_DRUGS}
_DISCOURAGED_PAIRS = {frozenset(p) for p in DISCOURAGED_COMBINATIONS}

_efficacy_artifact = None
_side_effect_artifact = None


def _load_artifacts():
    global _efficacy_artifact, _side_effect_artifact
    if _efficacy_artifact is None:
        _efficacy_artifact = joblib.load(ARTIFACTS_DIR / "efficacy_model.joblib")
    if _side_effect_artifact is None:
        _side_effect_artifact = joblib.load(ARTIFACTS_DIR / "side_effect_model.joblib")
    return _efficacy_artifact, _side_effect_artifact


class UnresolvedCompoundError(Exception):
    pass


class UnsupportedCombinationError(Exception):
    pass


async def _load_compounds(names: list[str]) -> dict[str, ResolvedCompound]:
    async with async_session_factory() as session:
        rows = (await session.scalars(select(ResolvedCompound).where(ResolvedCompound.name.in_(names)))).all()
    found = {r.name: r for r in rows}
    missing = [n for n in names if n not in found]
    if missing:
        raise UnresolvedCompoundError(
            f"Not in resolved_compounds cache (run populate_cache.py, or these aren't in the "
            f"registry yet): {missing}"
        )
    return found


def check_discouraged(names: list[str]) -> tuple[bool, Optional[str]]:
    if len(names) != 2:
        return False, None
    classes = frozenset(_NAME_TO_CLASS[n] for n in names)
    if classes in _DISCOURAGED_PAIRS:
        return True, (
            f"{names[0]} + {names[1]} combines {' + '.join(c.value for c in classes)} — "
            "dual RAAS blockade, associated with elevated hyperkalemia/renal-impairment risk. "
            "This is a hard-coded clinical rule (see drug_registry.py DISCOURAGED_COMBINATIONS), "
            "not a model output — see DECISIONS.md #3 for why this isn't a statistical prediction."
        )
    return False, None


def _potency_confidence(compounds: list[ResolvedCompound]) -> float:
    """Heuristic, disclosed confidence signal based on how much bioactivity
    evidence backs each component — NOT a calibrated probability. Capped
    contribution per compound at 5 records so one well-studied drug can't
    fully offset a sparse one. See Critical Engineering Decision #8."""
    scores = [min(c.n_valid_potency_records, 5) / 5 for c in compounds]
    return round(sum(scores) / len(scores), 2) if scores else 0.0


async def predict_efficacy(compound_names: list[str], patient: PatientProfile) -> dict:
    if len(compound_names) not in (1, 2):
        raise UnsupportedCombinationError(
            f"Only single drugs or 2-drug combinations are supported (trained data doesn't "
            f"cover {len(compound_names)}-drug combinations) — got {compound_names}"
        )

    compounds_by_name = await _load_compounds(compound_names)
    compounds = [compounds_by_name[n] for n in compound_names]

    # Potency z-scores need the whole registry's class statistics, not just
    # the requested compound(s) — reload full cache for that context.
    async with async_session_factory() as session:
        rows = (await session.scalars(select(ResolvedCompound))).all()
    full_cache = {r.name: r for r in rows}
    z_scores = compute_potency_z_scores(full_cache)

    classes = [_NAME_TO_CLASS[n] for n in compound_names]
    row = {"n_drugs": len(compound_names)}
    is_discouraged, discouraged_reason = check_discouraged(compound_names)
    row["discouraged"] = int(is_discouraged)
    for c in ALL_CLASSES:
        row[f"has_{c.value}"] = 1 if c in classes else 0
    row["mean_potency_z"] = sum(z_scores[n] for n in compound_names) / len(compound_names)
    row.update(_rdkit_mean(compounds))

    efficacy_artifact, _ = _load_artifacts()
    X = pd.DataFrame([row])[efficacy_artifact["feature_cols"]]
    raw_pred = efficacy_artifact["model"].predict(X)[0]
    raw_systolic, raw_diastolic = float(raw_pred[0]), float(raw_pred[1])

    # Explicit, disclosed patient-covariate multiplier — applied OUTSIDE
    # the model, per Critical Engineering Decision #5. Never baked into
    # training.
    renal_factor = patient.renal_adjustment_factor()
    adj_systolic = raw_systolic * renal_factor
    adj_diastolic = raw_diastolic * renal_factor

    confidence = _potency_confidence(compounds)

    deltas = [
        DiseaseParameterDelta(
            parameter="systolic_bp",
            baseline_value=patient.baseline.systolic_bp,
            predicted_value=patient.baseline.systolic_bp - adj_systolic,
            predicted_delta=-adj_systolic,
            confidence=confidence,
        ),
        DiseaseParameterDelta(
            parameter="diastolic_bp",
            baseline_value=patient.baseline.diastolic_bp,
            predicted_value=patient.baseline.diastolic_bp - adj_diastolic,
            predicted_delta=-adj_diastolic,
            confidence=confidence,
        ),
    ]

    return {
        "deltas": deltas,
        "raw_model_output_mmHg": {"systolic": round(raw_systolic, 2), "diastolic": round(raw_diastolic, 2)},
        "renal_adjustment_factor": renal_factor,
        "discouraged_combination": is_discouraged,
        "discouraged_reason": discouraged_reason,
        "confidence": confidence,
    }


async def predict_side_effects(compound_names: list[str], top_k: int = 10) -> dict:
    compounds_by_name = await _load_compounds(compound_names)
    compounds = [compounds_by_name[n] for n in compound_names]

    _, se_artifact = _load_artifacts()
    vocabulary = se_artifact["vocabulary"]

    per_compound_probs = []
    for comp in compounds:
        row = {"mean_potency": comp.mean_potency if comp.mean_potency is not None else float("nan")}
        cls = _NAME_TO_CLASS[comp.name]
        for c in ALL_CLASSES:
            row[f"has_{c.value}"] = 1 if c == cls else 0
        row.update(_rdkit_mean([comp]))
        X = pd.DataFrame([row])[se_artifact["feature_cols"]]

        probs = {}
        for label_idx, estimator in enumerate(se_artifact["model"].estimators_):
            classes_ = getattr(estimator, "classes_", [0, 1])
            proba = estimator.predict_proba(X)[0]
            p1 = proba[list(classes_).index(1)] if 1 in classes_ else 0.0
            probs[vocabulary[label_idx]] = float(p1)
        per_compound_probs.append(probs)

    # Combination side effects = element-wise max across components — a
    # disclosed simplification (real combo-level SIDER data doesn't
    # exist; see DECISIONS.md #1 task-design note). Not a learned
    # interaction, just "assume the higher individual risk carries over."
    combined = {se: max(p.get(se, 0.0) for p in per_compound_probs) for se in vocabulary}
    top = dict(sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k])

    return {
        "side_effect_probabilities": top,
        "note": "Combination probabilities are the element-wise max of each compound's "
                "individual prediction, not a learned interaction — no real combination-level "
                "side-effect data exists to train on. See DECISIONS.md.",
    }
