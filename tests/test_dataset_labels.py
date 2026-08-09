"""Pins down the efficacy label formula from DECISIONS.md #1 — this is the
single biggest scientific-honesty decision in the project, so a silent
change to ALPHA, the class BP table, or the combination discount should
fail a test, not just get noticed by chance during a demo."""

import pytest

from app.db.models import ResolvedCompound
from app.models.dataset import (
    ALPHA,
    CLASS_BP_TABLE,
    TWO_DRUG_ADDITIVITY_DISCOUNT,
    _adjusted_delta,
    compute_potency_z_scores,
)
from app.schemas.compound import DrugClass


def make_compound(name, mean_potency=None) -> ResolvedCompound:
    return ResolvedCompound(name=name, drug_class="x", gene_symbol="x", mean_potency=mean_potency)


class TestClassBPTable:
    """Verified against Law MR, Morris JK, Wald NJ. BMJ 2009;338:b1665 —
    see DECISIONS.md #1. These are the numbers a viva question would
    target directly."""

    @pytest.mark.parametrize("cls,systolic,diastolic", [
        (DrugClass.THIAZIDE_DIURETIC, 8.8, 4.4),
        (DrugClass.BETA_BLOCKER, 9.2, 6.7),
        (DrugClass.ACE_INHIBITOR, 8.5, 4.7),
        (DrugClass.ARB, 10.3, 5.7),
        (DrugClass.CALCIUM_CHANNEL_BLOCKER, 8.8, 5.9),
    ])
    def test_values_match_source(self, cls, systolic, diastolic):
        assert CLASS_BP_TABLE[cls]["systolic"] == systolic
        assert CLASS_BP_TABLE[cls]["diastolic"] == diastolic

    def test_all_five_classes_present(self):
        assert len(CLASS_BP_TABLE) == 5


class TestAdjustedDelta:
    def test_zero_z_score_returns_class_baseline_unchanged(self):
        delta = _adjusted_delta(DrugClass.ACE_INHIBITOR, potency_z=0.0)
        assert delta["systolic"] == CLASS_BP_TABLE[DrugClass.ACE_INHIBITOR]["systolic"]
        assert delta["diastolic"] == CLASS_BP_TABLE[DrugClass.ACE_INHIBITOR]["diastolic"]

    def test_positive_z_score_increases_delta(self):
        baseline = CLASS_BP_TABLE[DrugClass.ARB]["systolic"]
        delta = _adjusted_delta(DrugClass.ARB, potency_z=1.0)
        assert delta["systolic"] == pytest.approx(baseline * (1 + ALPHA))

    def test_z_score_is_clipped_at_plus_minus_2(self):
        """A wildly potent outlier shouldn't be allowed to blow past a
        +/-2 std-dev adjustment — this is the deliberate bound from
        DECISIONS.md #1, not an accident of the math."""
        delta_at_2 = _adjusted_delta(DrugClass.ARB, potency_z=2.0)
        delta_at_100 = _adjusted_delta(DrugClass.ARB, potency_z=100.0)
        assert delta_at_2["systolic"] == delta_at_100["systolic"]

    def test_adjustment_never_exceeds_alpha_times_two(self):
        for cls, table in CLASS_BP_TABLE.items():
            delta = _adjusted_delta(cls, potency_z=100.0)
            max_expected = table["systolic"] * (1 + ALPHA * 2)
            assert delta["systolic"] == pytest.approx(max_expected)


class TestCombinationDiscount:
    def test_discount_is_less_than_one(self):
        """Combining two drugs must be sub-additive, not a naive doubling
        — this is the empirical Law/Morris/Wald finding this project
        deliberately encodes rather than ignores."""
        assert 0 < TWO_DRUG_ADDITIVITY_DISCOUNT < 1

    def test_discount_matches_source_ratio(self):
        # From the diastolic worked example: 1 drug 4.7mmHg -> 2 drugs
        # 8.9mmHg observed vs 9.4mmHg naive sum.
        assert TWO_DRUG_ADDITIVITY_DISCOUNT == pytest.approx(8.9 / 9.4, abs=0.001)


class TestPotencyZScores:
    def test_missing_potency_gets_zero_z_not_fabricated_value(self):
        """Critical Engineering Decision #4: missing potency data stays
        missing. z=0 means 'assume class average', not 'we made up a
        number for this compound'."""
        compounds = {
            "drug_a": make_compound("drug_a", mean_potency=8.0),
            "drug_b": make_compound("drug_b", mean_potency=None),
        }
        # Both need to be in HYPERTENSION_DRUGS for the by-class grouping
        # to find them — patch via the real registry names instead.
        from app.core.drug_registry import HYPERTENSION_DRUGS

        real_names = [d["name"] for d in HYPERTENSION_DRUGS if d["drug_class"].value == "thiazide_diuretic"]
        assert len(real_names) >= 1
        missing_compounds = {name: make_compound(name, mean_potency=None) for name in real_names}
        z_scores = compute_potency_z_scores(missing_compounds)
        for name in real_names:
            assert z_scores[name] == 0.0

    def test_known_potency_produces_nonzero_z_when_class_has_variance(self):
        from app.core.drug_registry import HYPERTENSION_DRUGS

        ace_names = [d["name"] for d in HYPERTENSION_DRUGS if d["drug_class"].value == "ace_inhibitor"]
        compounds = {
            ace_names[0]: make_compound(ace_names[0], mean_potency=9.0),
            ace_names[1]: make_compound(ace_names[1], mean_potency=7.0),
            ace_names[2]: make_compound(ace_names[2], mean_potency=8.0),
        }
        z_scores = compute_potency_z_scores(compounds)
        assert z_scores[ace_names[0]] > 0
        assert z_scores[ace_names[1]] < 0
