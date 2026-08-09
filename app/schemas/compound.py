"""Schemas for compounds/drugs as input to the pipeline, and the
canonical drug-class taxonomy for the hypertension domain."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DrugClass(str, Enum):
    ACE_INHIBITOR = "ace_inhibitor"
    ARB = "arb"  # angiotensin receptor blocker
    BETA_BLOCKER = "beta_blocker"
    CALCIUM_CHANNEL_BLOCKER = "calcium_channel_blocker"
    THIAZIDE_DIURETIC = "thiazide_diuretic"
    UNKNOWN = "unknown"


class CompoundQuery(BaseModel):
    """What the LLM parsing node extracts from free text for a single
    compound mentioned by the user."""

    raw_name: str = Field(..., description="Name exactly as the user typed it")
    resolved_name: Optional[str] = None
    drug_class: Optional[DrugClass] = None
    chembl_id: Optional[str] = None
    pubchem_cid: Optional[int] = None
    dose_mg: Optional[float] = None
    notes: Optional[str] = Field(None, description="Anything ambiguous the resolver should flag")


class SimulationRequest(BaseModel):
    """Top-level request object once parsing is complete: patient +
    one or more compounds (a 'combination' is just len(compounds) > 1)."""

    compounds: list[CompoundQuery]
    patient_description_raw: str
    question_intent: Optional[str] = Field(
        None, description="e.g. 'efficacy', 'side_effects', 'compare_to_standard', 'general'"
    )
