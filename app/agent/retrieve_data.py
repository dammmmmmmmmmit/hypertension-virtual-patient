"""
retrieve_data node: pulls mechanism-of-action context for each resolved
compound from the Qdrant RAG index built by
app/ingestion/build_rag_index.py.

Uses a semantic query (FastEmbed, via qdrant-client's `.query()`
convenience method) rather than an exact metadata filter — this is a
real retrieval step, not a hardcoded dict lookup, even though for an
exactly-resolved registry drug name the top hit will trivially be that
drug's own document. Keeping it as genuine retrieval (rather than a
lookup table) means this same node also degrades sensibly if the
registry grows past what fits in a hand-written dict.
"""

from qdrant_client import QdrantClient

from app.agent.state import AgentState, MechanismContext
from app.core.settings import settings
from app.ingestion.build_rag_index import COLLECTION_NAME

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


async def retrieve_data(state: AgentState) -> AgentState:
    if not state.get("all_resolved", False):
        # Nothing to retrieve mechanism context for if entity resolution
        # already failed — resolve_entities' unresolved_names already
        # explains why; don't spend an LLM/vector-search round-trip here.
        state["mechanism_contexts"] = []
        return state

    client = _get_client()
    contexts: list[MechanismContext] = []

    for compound in state["compounds"]:
        if not compound.resolved_name:
            continue
        hits = client.query(collection_name=COLLECTION_NAME, query_text=compound.resolved_name, limit=1)
        if hits:
            payload = hits[0].metadata
            contexts.append(
                MechanismContext(
                    drug_name=payload["drug_name"],
                    mechanism_text=payload["mechanism_text"],
                    source=payload["source"],
                )
            )

    state["mechanism_contexts"] = contexts

    fallback_used = [c["drug_name"] for c in contexts if c["source"] == "class_level_fallback"]
    if fallback_used:
        limitations = state.get("limitations", [])
        limitations.append(
            f"No ChEMBL-recorded mechanism-of-action text for {', '.join(fallback_used)} — "
            "the mechanism description for these is inferred from drug-class pharmacology, "
            "not a compound-specific literature record."
        )
        state["limitations"] = limitations

    return state
