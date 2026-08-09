"""
Trains the two ML tasks (see DECISIONS.md #3 for why there are two, not
three) and saves them to app/models/artifacts/.

Both datasets are tiny (60 rows efficacy, 14 rows side-effects) — this is
disclosed, not hidden. Evaluation uses leave-one-out CV (the only honest
option at this N; a held-out test split would be too small to mean
anything) and reports plain MAE / per-label accuracy, no attempt to dress
up small-sample metrics as more than they are.

Run: uv run python -m app.models.train
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

EFFICACY_FEATURE_COLS = [
    "n_drugs",
    "discouraged",
    "has_ace_inhibitor",
    "has_arb",
    "has_beta_blocker",
    "has_calcium_channel_blocker",
    "has_thiazide_diuretic",
    "mean_potency_z",
    "mean_molecular_weight",
    "mean_logp",
    "mean_tpsa",
    "mean_h_bond_donors",
    "mean_h_bond_acceptors",
    "mean_rotatable_bonds",
    "mean_aromatic_rings",
]
EFFICACY_TARGET_COLS = ["systolic_delta", "diastolic_delta"]

SE_FEATURE_COLS = [
    "mean_potency",
    "has_ace_inhibitor",
    "has_arb",
    "has_beta_blocker",
    "has_calcium_channel_blocker",
    "has_thiazide_diuretic",
    "mean_molecular_weight",
    "mean_logp",
    "mean_tpsa",
    "mean_h_bond_donors",
    "mean_h_bond_acceptors",
    "mean_rotatable_bonds",
    "mean_aromatic_rings",
]


def _make_lgbm_regressor() -> lgb.LGBMRegressor:
    # Tiny dataset -> shallow, heavily regularized trees. Defaults would
    # overfit instantly at n=60 with 15 features.
    return lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        num_leaves=7,
        min_child_samples=3,
        learning_rate=0.1,
        verbosity=-1,
    )


def _make_lgbm_classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=50,
        max_depth=2,
        num_leaves=3,
        min_child_samples=2,
        learning_rate=0.1,
        verbosity=-1,
    )


def train_efficacy_model() -> dict:
    df = pd.read_csv(PROCESSED_DIR / "efficacy_dataset.csv")
    df["discouraged"] = df["discouraged"].astype(int)
    X = df[EFFICACY_FEATURE_COLS]
    y = df[EFFICACY_TARGET_COLS]

    # Leave-one-out CV for an honest small-N error estimate before fitting
    # the final model on all rows.
    loo = LeaveOneOut()
    preds = np.zeros_like(y.values, dtype=float)
    for train_idx, test_idx in loo.split(X):
        model = MultiOutputRegressor(_make_lgbm_regressor())
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = model.predict(X.iloc[test_idx])

    mae_systolic = mean_absolute_error(y["systolic_delta"], preds[:, 0])
    mae_diastolic = mean_absolute_error(y["diastolic_delta"], preds[:, 1])

    # Fit final model on the full dataset for actual inference use.
    final_model = MultiOutputRegressor(_make_lgbm_regressor())
    final_model.fit(X, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "feature_cols": EFFICACY_FEATURE_COLS, "target_cols": EFFICACY_TARGET_COLS},
                ARTIFACTS_DIR / "efficacy_model.joblib")

    metrics = {
        "n_rows": len(df),
        "eval_method": "leave-one-out CV",
        "mae_systolic_mmHg": round(mae_systolic, 3),
        "mae_diastolic_mmHg": round(mae_diastolic, 3),
        "note": "Labels are semi-synthetic (class-anchored + disclosed heuristic adjustment), "
                "NOT real per-compound clinical outcomes. Low LOO-CV error here reflects the "
                "model recovering a smooth, low-noise constructed function, not validated "
                "real-world predictive accuracy. See DECISIONS.md #1.",
    }
    return metrics


def train_side_effect_model() -> dict:
    df = pd.read_csv(PROCESSED_DIR / "side_effect_dataset.csv")
    se_cols = [c for c in df.columns if c.startswith("se__")]
    vocabulary = [c[len("se__"):] for c in se_cols]

    X = df[SE_FEATURE_COLS]
    y = df[se_cols]

    n = len(df)
    loo = LeaveOneOut()
    preds = np.zeros(y.shape, dtype=int)
    for train_idx, test_idx in loo.split(X):
        y_train = y.iloc[train_idx]
        # Labels that are constant (all-0 or all-1) across the training
        # fold can't be fit by a classifier — predict the constant
        # directly rather than erroring, and skip only those columns'
        # models per fold.
        model = MultiOutputClassifier(_make_lgbm_classifier())
        varying_cols = [c for c in se_cols if y_train[c].nunique() > 1]
        constant_cols = [c for c in se_cols if c not in varying_cols]

        if varying_cols:
            model.fit(X.iloc[train_idx], y_train[varying_cols])
            varying_preds = model.predict(X.iloc[test_idx])
        else:
            varying_preds = np.zeros((1, 0), dtype=int)

        row_pred = np.zeros(len(se_cols), dtype=int)
        for i, c in enumerate(varying_cols):
            row_pred[se_cols.index(c)] = varying_preds[0, i]
        for c in constant_cols:
            row_pred[se_cols.index(c)] = int(y_train[c].iloc[0])
        preds[test_idx[0]] = row_pred

    accuracy_per_label = [accuracy_score(y[c], preds[:, i]) for i, c in enumerate(se_cols)]
    mean_accuracy = float(np.mean(accuracy_per_label))

    # Final model on full data
    final_model = MultiOutputClassifier(_make_lgbm_classifier())
    final_model.fit(X, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final_model, "feature_cols": SE_FEATURE_COLS, "vocabulary": vocabulary},
        ARTIFACTS_DIR / "side_effect_model.joblib",
    )

    metrics = {
        "n_rows": n,
        "n_labels": len(se_cols),
        "eval_method": "leave-one-out CV (n=14 — too small for a held-out split)",
        "mean_per_label_accuracy": round(mean_accuracy, 3),
        "note": "14 samples, 30 labels — this is a genuinely small-data problem. Per-label "
                "accuracy is inflated by labels that are near-constant across the registry "
                "(e.g. a side effect reported for 13/14 drugs is 'accurate' even predicting "
                "the majority class). Treat this model as a smoothed lookup over structural "
                "similarity, not a validated QSAR model. Do not oversell this number.",
    }
    return metrics


def main():
    print("=== Training efficacy model ===")
    efficacy_metrics = train_efficacy_model()
    print(json.dumps(efficacy_metrics, indent=2))

    print("\n=== Training side-effect model ===")
    se_metrics = train_side_effect_model()
    print(json.dumps(se_metrics, indent=2))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "training_metrics.json").write_text(
        json.dumps({"efficacy": efficacy_metrics, "side_effects": se_metrics}, indent=2)
    )
    print(f"\nSaved models + metrics to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
