"""
Core schemas for the virtual patient and the disease parameters we track.

Scope (Day 1 decision): hypertension only. Every field here should map to
something either (a) clinically standard and easy to source baseline
distributions for, or (b) directly usable as a feature/adjustment factor
in the prediction model. Resist the urge to add fields "for realism" that
don't feed the model or the report — they just create hallucination risk
in the LLM layer later.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


class Comorbidity(str, Enum):
    """Kept deliberately small — each one should have a defined effect
    on the adjustment logic in Step 1. Don't add one without also adding
    its adjustment rule."""

    NONE = "none"
    TYPE2_DIABETES = "type2_diabetes"
    CHRONIC_KIDNEY_DISEASE = "chronic_kidney_disease"
    HEART_FAILURE = "heart_failure"
    ASTHMA_COPD = "asthma_copd"  # relevant: beta-blocker caution


class DiseaseParameters(BaseModel):
    """The 'physiology' state we simulate. Baseline values before any
    drug is applied. All units are standard clinical units."""

    systolic_bp: float = Field(..., ge=70, le=250, description="mmHg")
    diastolic_bp: float = Field(..., ge=40, le=150, description="mmHg")
    heart_rate: float = Field(..., ge=30, le=200, description="bpm")
    serum_potassium: float = Field(4.2, ge=2.5, le=7.0, description="mmol/L")
    egfr: float = Field(
        90.0, ge=5, le=140, description="estimated glomerular filtration rate, mL/min/1.73m2 (renal function proxy)"
    )

    @field_validator("diastolic_bp")
    @classmethod
    def diastolic_lower_than_systolic(cls, v, info):
        systolic = info.data.get("systolic_bp")
        if systolic is not None and v >= systolic:
            raise ValueError("diastolic_bp must be lower than systolic_bp")
        return v


class PatientProfile(BaseModel):
    """A virtual patient. This is the object the LLM parsing node
    populates from natural language, with sane clinical defaults for
    anything the user doesn't specify (and the report should say what
    was defaulted vs. what was stated)."""

    age: int = Field(..., ge=18, le=100)
    sex: Sex
    weight_kg: float = Field(..., ge=30, le=250)
    baseline: DiseaseParameters
    comorbidities: list[Comorbidity] = Field(default_factory=lambda: [Comorbidity.NONE])
    current_medications: list[str] = Field(default_factory=list)

    def renal_adjustment_factor(self) -> float:
        """Simplified exposure adjustment: reduced renal clearance ->
        higher effective drug exposure. This is a deliberately simple
        stand-in for real PK modeling — must be disclosed as such in
        the final report, not presented as validated pharmacokinetics."""
        if self.baseline.egfr >= 90:
            return 1.0
        elif self.baseline.egfr >= 60:
            return 1.1
        elif self.baseline.egfr >= 30:
            return 1.3
        else:
            return 1.6


class DiseaseParameterDelta(BaseModel):
    """Predicted change in a disease parameter after drug(s) applied.
    This is what the ML model outputs, per parameter."""

    parameter: str
    baseline_value: float
    predicted_value: float
    predicted_delta: float
    confidence: Optional[float] = Field(None, ge=0, le=1)
    