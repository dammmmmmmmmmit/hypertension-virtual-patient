"""
resolve_entities node: matches each parsed CompoundQuery.raw_name against
the registry, using the cached resolved_compounds data (never re-hitting
ChEMBL live per query — that's what the Postgres cache from Week 1 is for).

Design per the continuation brief: "on ambiguous/misspelled names,
populate CompoundQuery.notes rather than silently guessing." This means
an exact (case-insensitive) name match resolves fully; anything else
gets a suggestion written to `notes` and is left UNRESOLVED (chembl_id
stays None) rather than auto-corrected and silently proceeded with. A
wrong silent guess in a clinical-adjacent tool is worse than asking the
user to confirm.
"""

import difflib

from sqlalchemy import select

from app.agent.state import AgentState
from app.core.drug_registry import HYPERTENSION_DRUGS
from app.db.models import ResolvedCompound
from app.db.session import async_session_factory
from app.schemas.compound import DrugClass

_REGISTRY_NAMES = [d["name"] for d in HYPERTENSION_DRUGS]
_NAME_TO_CLASS = {d["name"]: d["drug_class"] for d in HYPERTENSION_DRUGS}


async def resolve_entities(state: AgentState) -> AgentState:
    compounds = state.get("compounds", [])

    async with async_session_factory() as session:
        cache_rows = (await session.scalars(select(ResolvedCompound))).all()
    cache_by_name = {r.name: r for r in cache_rows}

    unresolved_names: list[str] = []

    for compound in compounds:
        query_name = compound.raw_name.strip().lower()

        exact_match = next((n for n in _REGISTRY_NAMES if n == query_name), None)

        if exact_match:
            cached = cache_by_name.get(exact_match)
            compound.resolved_name = exact_match
            compound.drug_class = _NAME_TO_CLASS[exact_match]
            compound.chembl_id = cached.parent_chembl_id if cached else None
            compound.pubchem_cid = cached.pubchem_cid if cached else None
            if cached is None:
                compound.notes = (
                    f"'{exact_match}' is in the registry but hasn't been ingested into "
                    f"resolved_compounds yet — run populate_cache.py."
                )
                unresolved_names.append(compound.raw_name)
            continue

        # No exact match — suggest, don't guess.
        close = difflib.get_close_matches(query_name, _REGISTRY_NAMES, n=1, cutoff=0.6)
        if close:
            compound.notes = (
                f"'{compound.raw_name}' not found in the registry. Did you mean '{close[0]}'? "
                f"Not auto-resolved — confirm before proceeding."
            )
        else:
            compound.notes = (
                f"'{compound.raw_name}' not found in the registry and no close match exists. "
                f"This project only covers {len(_REGISTRY_NAMES)} hypertension drugs across 5 classes "
                f"(see drug_registry.py) — it cannot resolve compounds outside that set."
            )
        compound.drug_class = DrugClass.UNKNOWN
        unresolved_names.append(compound.raw_name)

    state["compounds"] = compounds
    state["all_resolved"] = len(unresolved_names) == 0
    state["unresolved_names"] = unresolved_names
    return state
