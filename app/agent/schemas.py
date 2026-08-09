"""
Schemas specific to the agent's parse step — distinct from the core
domain schemas in app/schemas/, which stay LLM-agnostic on purpose.

PatientProfile (app/schemas/patient.py) requires baseline vitals with no
defaults at the schema level — by design, per that file's docstring:
"sane clinical defaults for anything the user doesn't specify (and the
report should say what was defaulted vs. what was stated)". Pydantic
itself can't express "this field was defaulted, not stated", so we track
that separately here rather than bending the domain schema to fit a
parsing concern.
"""

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.compound import CompoundQuery
from app.schemas.patient import PatientProfile


class ParsedPatient(BaseModel):
    patient: PatientProfile
    defaulted_fields: list[str] = Field(
        default_factory=list,
        description="Names of PatientProfile fields the LLM had to default because the user's "
                     "text didn't state them (e.g. 'baseline.heart_rate', 'weight_kg') — the "
                     "report must disclose these, not present them as patient-stated facts.",
    )


class ParseResult(BaseModel):
    """What `parse_query` extracts from the raw natural-language input.
    Always produced by the fine-tuned model (see app/agent/parse_query.py
    and app/agent/local_finetuned_model.py) — when a caller already
    supplies a real PatientProfile (e.g. a structured form), parse_query
    keeps that and discards `parsed_patient` from this result rather than
    running a different, unverified extraction path. See DECISIONS.md #7."""

    compounds: list[CompoundQuery]
    parsed_patient: ParsedPatient
    question_intent: Optional[str] = Field(
        None, description="e.g. 'efficacy', 'side_effects', 'compare_to_standard', 'general'"
    )
