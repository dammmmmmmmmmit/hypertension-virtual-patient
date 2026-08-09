"""
retrieve_comparators node: surfaces 1-2 clinically standard drugs/combos
relevant to what the user asked about, from drug_registry.py's
STANDARD_COMBINATIONS — used by generate_report's "comparison vs existing
standard drugs/combos" section. Discouraged-combination flagging is
handled by predict.py (it's tied to the prediction itself, not a
comparator suggestion) — this node only surfaces STANDARD_COMBINATIONS.
"""

from app.agent.state import AgentState, ComparatorInfo
from app.core.drug_registry import HYPERTENSION_DRUGS, STANDARD_COMBINATIONS
from app.schemas.compound import DrugClass

_DRUGS_BY_CLASS: dict[DrugClass, list[str]] = {}
for _d in HYPERTENSION_DRUGS:
    _DRUGS_BY_CLASS.setdefault(_d["drug_class"], []).append(_d["name"])


def _pick_example(cls: DrugClass, exclude: set[str]) -> str:
    candidates = [n for n in _DRUGS_BY_CLASS.get(cls, []) if n not in exclude]
    return (candidates or _DRUGS_BY_CLASS.get(cls, ["(none registered)"]))[0]


async def retrieve_comparators(state: AgentState) -> AgentState:
    if not state.get("all_resolved", False):
        state["comparators"] = []
        return state

    requested_names = {c.resolved_name for c in state["compounds"] if c.resolved_name}
    requested_classes = {c.drug_class for c in state["compounds"] if c.drug_class}

    comparators: list[ComparatorInfo] = []
    for class_a, class_b in STANDARD_COMBINATIONS:
        if class_a not in requested_classes and class_b not in requested_classes:
            continue
        # Skip suggesting the exact combination already requested.
        if len(requested_classes) == 2 and requested_classes == {class_a, class_b}:
            continue

        drug_a = _pick_example(class_a, requested_names)
        drug_b = _pick_example(class_b, requested_names)
        comparators.append(
            ComparatorInfo(
                description=(
                    f"Standard combination: {class_a.value.replace('_', ' ')} + "
                    f"{class_b.value.replace('_', ' ')} (e.g. {drug_a} + {drug_b})"
                ),
                drug_names=[drug_a, drug_b],
            )
        )
        if len(comparators) >= 2:
            break

    state["comparators"] = comparators
    return state
