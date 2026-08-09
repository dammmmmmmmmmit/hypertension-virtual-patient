"""Formalizes the informal verification done during ChEMBL debugging
(losartan's Kb-fallback recovery, censored-value rejection) as real
pytest cases. See DECISIONS.md / potency_utils.py docstrings for the
reasoning these pin down."""

import math

import pytest

from app.ingestion.potency_utils import compute_pX, extract_best_potency


class TestComputePX:
    def test_exact_relation_nanomolar(self):
        # Losartan's real Kb record: 3.0 nM, exact relation -> pX ~ 8.52.
        # This is the actual case that recovered losartan's coverage.
        result = compute_pX(value="3.0", units="nM", relation="=")
        assert result == pytest.approx(8.52, abs=0.01)

    def test_none_relation_treated_as_exact(self):
        # ChEMBL sometimes omits standard_relation for a clean point estimate.
        result = compute_pX(value="100", units="nM", relation=None)
        assert result is not None

    @pytest.mark.parametrize("relation", [">", "<", ">=", "<="])
    def test_censored_relations_rejected(self, relation):
        """Critical Engineering Decision #3: censored values must never be
        treated as point estimates."""
        assert compute_pX(value="100", units="nM", relation=relation) is None

    def test_unrecognized_unit_rejected(self):
        assert compute_pX(value="100", units="ng/mL", relation="=") is None

    def test_missing_value_returns_none(self):
        assert compute_pX(value=None, units="nM", relation="=") is None

    def test_missing_units_returns_none(self):
        assert compute_pX(value="100", units=None, relation="=") is None

    def test_non_numeric_value_returns_none(self):
        assert compute_pX(value="not_a_number", units="nM", relation="=") is None

    def test_zero_or_negative_value_returns_none(self):
        assert compute_pX(value="0", units="nM", relation="=") is None
        assert compute_pX(value="-5", units="nM", relation="=") is None

    @pytest.mark.parametrize("units,factor", [
        ("M", 1.0),
        ("mM", 1e-3),
        ("uM", 1e-6),
        ("nM", 1e-9),
        ("pM", 1e-12),
    ])
    def test_unit_conversion_correctness(self, units, factor):
        result = compute_pX(value="1.0", units=units, relation="=")
        expected = round(-math.log10(1.0 * factor), 2)
        assert result == expected


class TestExtractBestPotency:
    def test_prefers_pchembl_value_when_present(self):
        activity = {"pchembl_value": "7.5", "standard_value": "999", "standard_units": "nM", "standard_relation": "="}
        assert extract_best_potency(activity) == 7.5

    def test_falls_back_to_manual_px_when_pchembl_missing(self):
        # This mirrors losartan's real record: no pchembl_value, but a
        # usable Kb with an exact relation.
        activity = {"pchembl_value": None, "standard_value": "3.0", "standard_units": "nM", "standard_relation": "="}
        assert extract_best_potency(activity) == pytest.approx(8.52, abs=0.01)

    def test_returns_none_when_both_unavailable(self):
        activity = {"pchembl_value": None, "standard_value": "3.0", "standard_units": "nM", "standard_relation": ">"}
        assert extract_best_potency(activity) is None

    def test_returns_none_for_completely_empty_record(self):
        assert extract_best_potency({}) is None

    def test_invalid_pchembl_value_falls_back(self):
        activity = {"pchembl_value": "not_a_float", "standard_value": "10", "standard_units": "nM", "standard_relation": "="}
        result = extract_best_potency(activity)
        assert result is not None
        assert result == pytest.approx(8.0, abs=0.01)
