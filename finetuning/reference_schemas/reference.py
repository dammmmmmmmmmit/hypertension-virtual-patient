"""
REFERENCE ONLY - reconstructed from the original Day-1 scaffold schemas.
Your actual app/agent/schemas.py has evolved since then (per your Week 2
report) and I don't have its current contents. This file exists only so
the synthetic data generator has something real to import and I can
actually test it. Match field names against YOUR real schema before
trusting the generator's output field names.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


class Comorbidity(str, Enum):
    NONE = "none"
    TYPE2_DIABETES = "type2_diabetes"
    CHRONIC_KIDNEY_DISEASE = "chronic_kidney_disease"
    HEART_FAILURE = "heart_failure"
    ASTHMA_COPD = "asthma_copd"


class DiseaseParameters(BaseModel):
    systolic_bp: float = Field(..., ge=70, le=250)
    diastolic_bp: float = Field(..., ge=40, le=150)
    heart_rate: float = Field(..., ge=30, le=200)
    serum_potassium: float = Field(4.2, ge=2.5, le=7.0)
    egfr: float = Field(90.0, ge=5, le=140)


class PatientProfile(BaseModel):
    age: int = Field(..., ge=18, le=100)
    sex: Sex
    weight_kg: float = Field(..., ge=30, le=250)
    baseline: DiseaseParameters
    comorbidities: list[Comorbidity] = Field(default_factory=lambda: [Comorbidity.NONE])
    current_medications: list[str] = Field(default_factory=list)


class DrugClass(str, Enum):
    ACE_INHIBITOR = "ace_inhibitor"
    ARB = "arb"
    BETA_BLOCKER = "beta_blocker"
    CALCIUM_CHANNEL_BLOCKER = "calcium_channel_blocker"
    THIAZIDE_DIURETIC = "thiazide_diuretic"
    UNKNOWN = "unknown"


class CompoundQuery(BaseModel):
    raw_name: str
    dose_mg: Optional[float] = None


class SimulationRequest(BaseModel):
    compounds: list[CompoundQuery]
    patient_description_raw: str
    question_intent: Optional[str] = None
