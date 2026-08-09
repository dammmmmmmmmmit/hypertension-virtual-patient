"""Request/response schemas for the low-level prediction API. These are
API-transport shapes, distinct from the core domain schemas in
app/schemas/ — this endpoint takes already-resolved drug names, it doesn't
parse natural language (that's the Week-2 agent's job)."""

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.patient import DiseaseParameterDelta, PatientProfile


class PredictionRequest(BaseModel):
    compound_names: list[str] = Field(..., min_length=1, max_length=2, description="Registry drug names, e.g. ['lisinopril'] or ['amlodipine', 'losartan']")
    patient: PatientProfile


class PredictionResponse(BaseModel):
    compound_names: list[str]
    disease_parameter_deltas: list[DiseaseParameterDelta]
    side_effect_probabilities: dict[str, float]
    discouraged_combination: bool
    discouraged_reason: Optional[str] = None
    renal_adjustment_factor: float
    limitations: list[str]


class SimulateRequest(BaseModel):
    """Request body for POST /simulate/stream — mirrors exactly what the
    Streamlit UI already passes to run_agent(raw_query, patient=patient):
    free-text compound(s)/question + a structured patient profile (never
    parsed from prose — see app/ui/streamlit_app.py's docstring on why)."""

    raw_query: str
    patient: PatientProfile
