"""
Wires the nodes from Section 3 of the continuation brief into a
LangGraph StateGraph:

parse_query -> resolve_entities -> [resolved?]
  no  -> generate_report (clarification path) -> END
  yes -> retrieve_data -> structure_features -> [still resolved / supported combo size?]
           no  -> generate_report (clarification path) -> END
           yes -> run_prediction -> retrieve_comparators -> generate_report -> END

The two conditional branches both short-circuit straight to
generate_report rather than running the remaining nodes uselessly (no
point retrieving comparators for a combination we never predicted
anything for) — generate_report itself already knows how to produce a
clarification-style report when `all_resolved` is False or `prediction`
is None (see generate_report.py).

Run: uv run python -m app.agent.graph "<some query>"  (smoke test —
requires a real ANTHROPIC_API_KEY in .env, not yet available in this
session, see DECISIONS.md)
"""

import asyncio
import sys
from typing import Optional

from langgraph.graph import END, StateGraph

from app.agent.comparators import retrieve_comparators
from app.agent.generate_report import generate_report
from app.agent.parse_query import parse_query
from app.agent.predict import run_prediction, structure_features
from app.agent.resolve_entities import resolve_entities
from app.agent.retrieve_data import retrieve_data
from app.agent.schemas import ParsedPatient
from app.agent.state import AgentState
from app.schemas.patient import PatientProfile


def _entities_resolved(state: AgentState) -> str:
    return "resolved" if state.get("all_resolved", False) else "unresolved"


def _features_structured(state: AgentState) -> str:
    return "resolved" if state.get("all_resolved", False) else "unresolved"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_query", parse_query)
    graph.add_node("resolve_entities", resolve_entities)
    graph.add_node("retrieve_data", retrieve_data)
    graph.add_node("structure_features", structure_features)
    graph.add_node("run_prediction", run_prediction)
    graph.add_node("retrieve_comparators", retrieve_comparators)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("parse_query")
    graph.add_edge("parse_query", "resolve_entities")

    graph.add_conditional_edges(
        "resolve_entities",
        _entities_resolved,
        {"resolved": "retrieve_data", "unresolved": "generate_report"},
    )
    graph.add_edge("retrieve_data", "structure_features")

    graph.add_conditional_edges(
        "structure_features",
        _features_structured,
        {"resolved": "run_prediction", "unresolved": "generate_report"},
    )
    graph.add_edge("run_prediction", "retrieve_comparators")
    graph.add_edge("retrieve_comparators", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


async def run_agent(raw_query: str, patient: Optional[PatientProfile] = None) -> AgentState:
    """If `patient` is given (Streamlit's structured form, per Section 10),
    parse_query only extracts compounds/intent from raw_query and skips
    LLM patient-field guessing entirely — see parse_query.py."""
    app = build_graph()
    initial_state: AgentState = {"raw_query": raw_query, "limitations": []}
    if patient is not None:
        initial_state["parsed_patient"] = ParsedPatient(patient=patient, defaulted_fields=[])
    return await app.ainvoke(initial_state)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What happens if I give lisinopril to a 60 year old male with baseline BP 150/95?"
    final_state = asyncio.run(run_agent(query))
    print(final_state.get("report"))
