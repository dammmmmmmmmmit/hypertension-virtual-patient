"""
structure_features + run_prediction nodes, combined in one module since
the actual feature-vector construction already lives inside
app/models/inference.py's predict_efficacy/predict_side_effects (built
and tested in Week 1) — there's no separate feature-building step to
duplicate here. `structure_features` in this file is a validation gate
(enough resolved compounds? a supported combination size?); the real
"structured feature vector" work happens where it was already verified.

Deliberate deviation from the Section 3 architecture diagram: the diagram
says run_prediction "calls FastAPI-served ML model", implying an HTTP
round-trip. This agent runs in the same process as the model artifacts
(app/models/artifacts/*.joblib) already loaded by inference.py, so it
calls those functions directly instead of making an HTTP request to its
own FastAPI app — avoids a pointless self-network-call and an extra
"is the API server running" runtime dependency for the agent to work.
The FastAPI layer (app/api/) still exists and wraps the same functions
for any external caller that only wants raw predictions without the
agent/report layer.
"""

from app.agent.state import AgentState
from app.models.inference import UnresolvedCompoundError, UnsupportedCombinationError, predict_efficacy, predict_side_effects


async def structure_features(state: AgentState) -> AgentState:
    """Validation gate: only single drugs or 2-drug combinations are
    supported (matches what the model was trained on — see DECISIONS.md
    #1, #3). Sets prediction=None and a limitation note if not."""
    compounds = [c for c in state.get("compounds", []) if c.resolved_name]

    if not state.get("all_resolved", False):
        return state  # resolve_entities already recorded why; nothing to do here

    if len(compounds) not in (1, 2):
        limitations = state.get("limitations", [])
        limitations.append(
            f"This tool only models single drugs or 2-drug combinations — "
            f"{len(compounds)} resolved compounds were given, so no prediction was run."
        )
        state["limitations"] = limitations
        state["prediction"] = None
        state["all_resolved"] = False  # short-circuits run_prediction/generate_report's happy path
    return state


async def run_prediction(state: AgentState) -> AgentState:
    if not state.get("all_resolved", False) or state.get("prediction") is not None and state["prediction"] is None:
        return state

    compounds = [c for c in state["compounds"] if c.resolved_name]
    names = [c.resolved_name for c in compounds]
    patient = state["parsed_patient"].patient

    try:
        efficacy = await predict_efficacy(names, patient)
        side_effects = await predict_side_effects(names)
    except (UnresolvedCompoundError, UnsupportedCombinationError) as e:
        limitations = state.get("limitations", [])
        limitations.append(f"Prediction could not be run: {e}")
        state["limitations"] = limitations
        state["prediction"] = None
        return state

    state["prediction"] = {
        "deltas": efficacy["deltas"],
        "raw_model_output_mmHg": efficacy["raw_model_output_mmHg"],
        "renal_adjustment_factor": efficacy["renal_adjustment_factor"],
        "confidence": efficacy["confidence"],
        "side_effect_probabilities": side_effects["side_effect_probabilities"],
    }

    limitations = state.get("limitations", [])
    if efficacy["discouraged_combination"]:
        state["discouraged_warning"] = efficacy["discouraged_reason"]
        limitations.append(efficacy["discouraged_reason"])
    if efficacy["confidence"] < 0.5:
        limitations.append(
            f"Low bioactivity evidence backing this prediction (confidence={efficacy['confidence']}) "
            "— treat the efficacy numbers with extra caution."
        )
    state["limitations"] = limitations
    return state
