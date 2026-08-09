"""LangGraph shared state, threaded through every node in app/agent/graph.py.

Design note: this is a plain TypedDict (LangGraph's standard state
pattern), not one of the Pydantic domain schemas — it's a bag of
intermediate results specific to one graph run, not a reusable domain
object. The Pydantic schemas (SimulationRequest, PatientProfile, etc.)
live INSIDE this state as values.
"""

from typing import Optional, TypedDict

from app.agent.schemas import ParsedPatient
from app.schemas.compound import CompoundQuery


class MechanismContext(TypedDict):
    drug_name: str
    mechanism_text: str
    source: str  # "chembl_mechanism" | "class_level_fallback" — see build_rag_index.py


class ComparatorInfo(TypedDict):
    description: str
    drug_names: list[str]


class TokenUsage(TypedDict):
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # Meaningful only for parse_query (the one node with an internal
    # retry loop, see local_finetuned_model.py's MAX_ATTEMPTS) — 1 for
    # every other LLM-calling node. input/output/total above are already
    # summed across all attempts, since a failed attempt is still real
    # GPU compute spent, not free to omit from an honest total.
    attempts: int


class AgentState(TypedDict, total=False):
    # Input
    raw_query: str

    # After parse_query
    compounds: list[CompoundQuery]
    parsed_patient: Optional[ParsedPatient]
    question_intent: Optional[str]

    # After resolve_entities — compounds list is updated in place with
    # resolved_name/drug_class/chembl_id/notes; this flag short-circuits
    # downstream nodes if anything couldn't be resolved.
    all_resolved: bool
    unresolved_names: list[str]

    # After retrieve_data
    mechanism_contexts: list[MechanismContext]

    # After run_prediction
    prediction: Optional[dict]

    # After retrieve_comparators
    comparators: list[ComparatorInfo]
    discouraged_warning: Optional[str]

    # After generate_report
    report: Optional[str]

    # Accumulated non-fatal issues surfaced across nodes, always shown in
    # the final report's caveats section (Critical Engineering Decision #6).
    limitations: list[str]

    # Keyed by node name — only parse_query and generate_report ever set
    # an entry (the only two nodes that call an LLM). Each node reads the
    # existing dict, adds its own key, and returns the whole thing, since
    # LangGraph's default per-key reducer is "last write wins" (a plain
    # TypedDict here, no custom Annotated merge reducer) — returning only
    # a fresh single-entry dict would silently drop earlier nodes' usage.
    token_usage: dict[str, TokenUsage]
