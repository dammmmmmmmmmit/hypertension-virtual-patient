"""PatientProfile / DiseaseParameters validator behavior."""

import pytest
from pydantic import ValidationError

from app.schemas.patient import Comorbidity, DiseaseParameters, PatientProfile, Sex


def make_baseline(**overrides) -> DiseaseParameters:
    defaults = dict(systolic_bp=150, diastolic_bp=95, heart_rate=78)
    defaults.update(overrides)
    return DiseaseParameters(**defaults)


def make_patient(**overrides) -> PatientProfile:
    defaults = dict(age=55, sex=Sex.MALE, weight_kg=80, baseline=make_baseline())
    defaults.update(overrides)
    return PatientProfile(**defaults)


class TestDiseaseParametersValidation:
    def test_valid_baseline(self):
        dp = make_baseline()
        assert dp.systolic_bp == 150
        assert dp.diastolic_bp == 95

    def test_diastolic_must_be_lower_than_systolic(self):
        with pytest.raises(ValidationError, match="diastolic_bp must be lower than systolic_bp"):
            make_baseline(systolic_bp=120, diastolic_bp=120)

    def test_diastolic_higher_than_systolic_rejected(self):
        with pytest.raises(ValidationError, match="diastolic_bp must be lower than systolic_bp"):
            make_baseline(systolic_bp=110, diastolic_bp=130)

    @pytest.mark.parametrize("field,bad_value", [
        ("systolic_bp", 60),   # below ge=70
        ("systolic_bp", 260),  # above le=250
        ("diastolic_bp", 30),  # below ge=40, and also < systolic issue avoided by testing bound alone
        ("heart_rate", 20),    # below ge=30
        ("heart_rate", 250),   # above le=200
    ])
    def test_out_of_range_values_rejected(self, field, bad_value):
        kwargs = {field: bad_value}
        with pytest.raises(ValidationError):
            make_baseline(**kwargs)

    def test_defaults_applied(self):
        dp = make_baseline()
        assert dp.serum_potassium == 4.2
        assert dp.egfr == 90.0


class TestPatientProfileValidation:
    def test_valid_patient(self):
        p = make_patient()
        assert p.age == 55
        assert p.comorbidities == [Comorbidity.NONE]
        assert p.current_medications == []

    @pytest.mark.parametrize("age", [17, 101])
    def test_age_bounds(self, age):
        with pytest.raises(ValidationError):
            make_patient(age=age)

    @pytest.mark.parametrize("weight", [20, 300])
    def test_weight_bounds(self, weight):
        with pytest.raises(ValidationError):
            make_patient(weight_kg=weight)

    def test_comorbidities_list_accepted(self):
        p = make_patient(comorbidities=[Comorbidity.TYPE2_DIABETES, Comorbidity.CHRONIC_KIDNEY_DISEASE])
        assert Comorbidity.CHRONIC_KIDNEY_DISEASE in p.comorbidities


class TestRenalAdjustmentFactor:
    """This is a deliberately simple stand-in for real PK modeling (see
    patient.py docstring) — pin down its exact threshold behavior so a
    future refactor can't silently change clinical-facing numbers."""

    @pytest.mark.parametrize("egfr,expected_factor", [
        (95, 1.0),   # normal
        (90, 1.0),   # boundary: >= 90
        (89.9, 1.1), # just below 90
        (60, 1.1),   # boundary: >= 60
        (59.9, 1.3), # just below 60
        (30, 1.3),   # boundary: >= 30
        (29.9, 1.6), # just below 30 -> severe
        (10, 1.6),
    ])
    def test_thresholds(self, egfr, expected_factor):
        patient = make_patient(baseline=make_baseline(egfr=egfr))
        assert patient.renal_adjustment_factor() == expected_factor
