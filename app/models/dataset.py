"""
Assembles the two training datasets described in DECISIONS.md #1-3:

1. Efficacy dataset — single drugs + 2-drug combinations, feature vector
   describing which classes are present + within-class potency deviation +
   aggregate structural descriptors, label = class-anchored BP delta with
   the disclosed potency adjustment and combination-additivity discount.
   This is a SEMI-SYNTHETIC label set, not real per-compound clinical
   outcomes — see DECISIONS.md #1 before trusting/extending this blindly.

2. Side-effect dataset — single drugs only (no real combo-level SIDER
   data exists), feature vector = structural descriptors + potency,
   label = multi-hot vector over the top-N most common PT-level side
   effects across the registry. Real labels (SIDER), small N (14 drugs
   with coverage) — see DECISIONS.md #1 task-design note on why this is
   evaluated with leave-one-out CV, not a held-out split.

Run: uv run python -m app.models.dataset  (prints both datasets' shapes
and writes them to data/processed/ for train.py to consume without
re-hitting the DB every time.)
"""

import asyncio
import itertools
import statistics
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from app.core.drug_registry import DISCOURAGED_COMBINATIONS, HYPERTENSION_DRUGS, STANDARD_COMBINATIONS
from app.db.models import ResolvedCompound
from app.db.session import async_session_factory
from app.schemas.compound import DrugClass

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Class-level standard-dose BP reduction, verified against Law MR, Morris JK,
# Wald NJ. BMJ 2009;338:b1665 (cross-checked NIHR HTA NBK62259) — see
# DECISIONS.md #1. Do not edit without re-verifying against the source.
CLASS_BP_TABLE: dict[DrugClass, dict[str, float]] = {
    DrugClass.THIAZIDE_DIURETIC: {"systolic": 8.8, "diastolic": 4.4},
    DrugClass.BETA_BLOCKER: {"systolic": 9.2, "diastolic": 6.7},
    DrugClass.ACE_INHIBITOR: {"systolic": 8.5, "diastolic": 4.7},
    DrugClass.ARB: {"systolic": 10.3, "diastolic": 5.7},
    DrugClass.CALCIUM_CHANNEL_BLOCKER: {"systolic": 8.8, "diastolic": 5.9},
}

# Small, hand-set, DISCLOSED heuristic bound on within-class potency
# adjustment — not fit from per-compound clinical data (none exists at
# this scope). See DECISIONS.md #1 "honesty caveat".
ALPHA = 0.08
POTENCY_Z_CLIP = 2.0

# Empirical 2-drug additivity discount, derived from the Law/Morris/Wald
# diastolic worked example (1 drug 4.7mmHg -> 2 drugs 8.9mmHg, 94.7% of the
# naive sum). Applied to both systolic and diastolic per the "roughly
# additive, mildly sub-additive" conclusion — see DECISIONS.md #1.
TWO_DRUG_ADDITIVITY_DISCOUNT = 0.947

ALL_CLASSES = [
    DrugClass.ACE_INHIBITOR,
    DrugClass.ARB,
    DrugClass.BETA_BLOCKER,
    DrugClass.CALCIUM_CHANNEL_BLOCKER,
    DrugClass.THIAZIDE_DIURETIC,
]


async def load_resolved_compounds() -> dict[str, ResolvedCompound]:
    async with async_session_factory() as session:
        rows = (await session.scalars(select(ResolvedCompound))).all()
    return {r.name: r for r in rows}


def compute_potency_z_scores(compounds: dict[str, ResolvedCompound]) -> dict[str, float]:
    """Within-class z-score of mean_potency. A drug with missing potency
    (the 3 thiazides) gets z=0 — i.e. "assume class-average", which is
    explicitly NOT the same as imputing a fabricated potency VALUE; we
    never invent a pchembl number, we just decline to say this compound
    deviates from its class in either direction. See DECISIONS.md #1."""
    by_class: dict[DrugClass, list[float]] = {}
    for drug in HYPERTENSION_DRUGS:
        comp = compounds.get(drug["name"])
        if comp and comp.mean_potency is not None:
            by_class.setdefault(drug["drug_class"], []).append(comp.mean_potency)

    z_scores: dict[str, float] = {}
    for drug in HYPERTENSION_DRUGS:
        name = drug["name"]
        comp = compounds.get(name)
        cls = drug["drug_class"]
        values = by_class.get(cls, [])
        if comp is None or comp.mean_potency is None or len(values) < 2:
            z_scores[name] = 0.0
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        z_scores[name] = 0.0 if std == 0 else (comp.mean_potency - mean) / std
    return z_scores


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rdkit_mean(compounds_subset: list[ResolvedCompound]) -> dict[str, float]:
    keys = ["molecular_weight", "logp", "tpsa", "h_bond_donors", "h_bond_acceptors", "rotatable_bonds", "aromatic_rings"]
    out = {}
    for k in keys:
        vals = [c.rdkit_descriptors[k] for c in compounds_subset if c.rdkit_descriptors]
        out[f"mean_{k}"] = statistics.mean(vals) if vals else float("nan")
    return out


def _adjusted_delta(cls: DrugClass, potency_z: float) -> dict[str, float]:
    base = CLASS_BP_TABLE[cls]
    factor = 1 + ALPHA * _clip(potency_z, -POTENCY_Z_CLIP, POTENCY_Z_CLIP)
    return {"systolic": base["systolic"] * factor, "diastolic": base["diastolic"] * factor}


def build_efficacy_rows(compounds: dict[str, ResolvedCompound], z_scores: dict[str, float]) -> list[dict]:
    rows = []
    discouraged_class_pairs = {frozenset(p) for p in DISCOURAGED_COMBINATIONS}
    standard_class_pairs = {frozenset(p) for p in STANDARD_COMBINATIONS}

    # Single-drug rows
    for drug in HYPERTENSION_DRUGS:
        name = drug["name"]
        comp = compounds.get(name)
        if comp is None:
            continue
        cls = drug["drug_class"]
        delta = _adjusted_delta(cls, z_scores[name])
        row = {
            "label": name,
            "n_drugs": 1,
            "discouraged": False,
            "systolic_delta": delta["systolic"],
            "diastolic_delta": delta["diastolic"],
        }
        for c in ALL_CLASSES:
            row[f"has_{c.value}"] = 1 if c == cls else 0
        row["mean_potency_z"] = z_scores[name]
        row.update(_rdkit_mean([comp]))
        rows.append(row)

    # 2-drug combination rows — every cross-class pair actually present in
    # STANDARD_COMBINATIONS or DISCOURAGED_COMBINATIONS, expanded to real
    # drug pairs (not every arbitrary cross-class pair — keeps the dataset
    # anchored to clinically meaningful combinations, per Section 9's own
    # framing of "standard combo" and "discouraged combo" as the two cases
    # to cover).
    all_class_pairs = list(standard_class_pairs | discouraged_class_pairs)
    for pair in all_class_pairs:
        cls_a, cls_b = tuple(pair)
        drugs_a = [d for d in HYPERTENSION_DRUGS if d["drug_class"] == cls_a]
        drugs_b = [d for d in HYPERTENSION_DRUGS if d["drug_class"] == cls_b]
        for da, db in itertools.product(drugs_a, drugs_b):
            name_a, name_b = da["name"], db["name"]
            comp_a, comp_b = compounds.get(name_a), compounds.get(name_b)
            if comp_a is None or comp_b is None:
                continue
            delta_a = _adjusted_delta(cls_a, z_scores[name_a])
            delta_b = _adjusted_delta(cls_b, z_scores[name_b])
            is_discouraged = pair in discouraged_class_pairs
            row = {
                "label": f"{name_a}+{name_b}",
                "n_drugs": 2,
                "discouraged": is_discouraged,
                "systolic_delta": (delta_a["systolic"] + delta_b["systolic"]) * TWO_DRUG_ADDITIVITY_DISCOUNT,
                "diastolic_delta": (delta_a["diastolic"] + delta_b["diastolic"]) * TWO_DRUG_ADDITIVITY_DISCOUNT,
            }
            for c in ALL_CLASSES:
                row[f"has_{c.value}"] = 1 if c in (cls_a, cls_b) else 0
            row["mean_potency_z"] = statistics.mean([z_scores[name_a], z_scores[name_b]])
            row.update(_rdkit_mean([comp_a, comp_b]))
            rows.append(row)

    return rows


def build_side_effect_rows(compounds: dict[str, ResolvedCompound], top_n: int = 30) -> tuple[list[dict], list[str]]:
    """Build the multi-label side-effect target vector.

    IMPORTANT — discovered by testing the deployed model, not by inspection:
    naively taking the N most-frequent PT terms selects boilerplate adverse
    events (headache, nausea, dizziness, rash...) that SIDER's label-derived
    data reports for nearly every drug (all 14/14 in this registry), because
    US drug labels routinely list these regardless of actual mechanism. A
    classifier trained on all-1 (or all-0) columns has nothing to learn and
    every prediction converges to the same generic list for every compound
    — verified live: atenolol and hydrochlorothiazide returned byte-identical
    top-10 side-effect predictions. Filtering to labels with real variance
    across the registry (neither near-universal nor near-absent) is what
    makes this task a genuine classification problem instead of trivia.
    """
    from collections import Counter

    counts: Counter = Counter()
    n_drugs_with_data = sum(1 for c in compounds.values() if c.side_effects)
    for comp in compounds.values():
        for se in comp.side_effects or []:
            counts[se] += 1

    min_count = 2
    max_count = max(min_count, n_drugs_with_data - 2)
    discriminative = {se: n for se, n in counts.items() if min_count <= n <= max_count}
    vocabulary = [name for name, _ in Counter(discriminative).most_common(top_n)]

    rows = []
    for drug in HYPERTENSION_DRUGS:
        name = drug["name"]
        comp = compounds.get(name)
        if comp is None or not comp.side_effects:
            continue  # genuine SIDER gap (enalapril) — excluded, not imputed
        row = {
            "label": name,
            "mean_potency": comp.mean_potency if comp.mean_potency is not None else float("nan"),
        }
        for c in ALL_CLASSES:
            row[f"has_{c.value}"] = 1 if c == drug["drug_class"] else 0
        row.update(_rdkit_mean([comp]))
        se_set = set(comp.side_effects)
        for se in vocabulary:
            row[f"se__{se}"] = 1 if se in se_set else 0
        rows.append(row)

    return rows, vocabulary


async def main():
    compounds = await load_resolved_compounds()
    z_scores = compute_potency_z_scores(compounds)

    efficacy_rows = build_efficacy_rows(compounds, z_scores)
    efficacy_df = pd.DataFrame(efficacy_rows)

    se_rows, vocabulary = build_side_effect_rows(compounds)
    se_df = pd.DataFrame(se_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    efficacy_df.to_csv(PROCESSED_DIR / "efficacy_dataset.csv", index=False)
    se_df.to_csv(PROCESSED_DIR / "side_effect_dataset.csv", index=False)
    (PROCESSED_DIR / "side_effect_vocabulary.txt").write_text("\n".join(vocabulary))

    print(f"Efficacy dataset: {efficacy_df.shape[0]} rows, {efficacy_df.shape[1]} cols")
    print(f"  singles: {(efficacy_df.n_drugs == 1).sum()}, combos: {(efficacy_df.n_drugs == 2).sum()}, "
          f"discouraged: {efficacy_df.discouraged.sum()}")
    print(f"Side-effect dataset: {se_df.shape[0]} rows, vocabulary size {len(vocabulary)}")
    print(f"  (excluded: enalapril — no SIDER coverage, see DECISIONS.md)")


if __name__ == "__main__":
    asyncio.run(main())
