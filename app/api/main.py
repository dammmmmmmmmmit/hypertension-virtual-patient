"""
FastAPI app wrapping the trained models (app/models/inference.py). This is
the low-level prediction endpoint — it expects already-resolved drug
names, not free text. The Week-2 LangGraph agent's `run_prediction` node
will call this (or the underlying inference functions directly).

Also exposes `/simulate/stream` — the full agent pipeline
(parse_query -> ... -> generate_report) as Server-Sent Events, one event
per real LangGraph node completion. This exists specifically for the
Next.js frontend's pipeline visualization: the brief requires the
animation timing to reflect REAL stage durations (parse_query ~35-40s,
generate_report ~90-130s, see DECISIONS.md #7), not a decorative client
timer. Streaming real `graph.astream()` node-completion events is what
makes that possible — a plain blocking POST would give the frontend
nothing to animate against except a guess. The Streamlit UI doesn't need
this (it calls `run_agent()` in-process, same Python runtime); Next.js
runs as a separate process and needs HTTP.

Run: uv run uvicorn app.api.main:app --reload
"""

import json

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agent.graph import build_graph
from app.agent.schemas import ParsedPatient
from app.api.schemas import PredictionRequest, PredictionResponse, SimulateRequest
from app.models.inference import UnresolvedCompoundError, UnsupportedCombinationError, predict_efficacy, predict_side_effects

app = FastAPI(
    title="Virtual Patient Drug-Response Simulator — Prediction API",
    description="Hypertension-only screening tool. Not clinical guidance. See README/DECISIONS.md.",
)

# Local dev only — the frontend runs on a different origin (localhost:3000)
# than this API (localhost:8000). Both are localhost-only in this project;
# tighten allow_origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standing disclosures every prediction response carries — see Critical
# Engineering Decisions #5, #6, #8 in the continuation brief. Keep these
# in sync with what the report generator (Week 2) says.
STANDARD_LIMITATIONS = [
    "This is a screening/demo tool, not clinical guidance.",
    "Efficacy predictions are anchored to published class-level trial data (Law/Morris/Wald, "
    "BMJ 2009) with a small, disclosed heuristic adjustment for compound-specific potency — "
    "not a model trained on real per-compound clinical outcomes. See DECISIONS.md #1.",
    "Patient covariate adjustment (renal function) is a simplified multiplier, not validated PK/PD modeling.",
    "Side-effect combination probabilities are the max of each drug's individual prediction, not a learned interaction.",
]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        efficacy_result = await predict_efficacy(request.compound_names, request.patient)
        se_result = await predict_side_effects(request.compound_names)
    except UnresolvedCompoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except UnsupportedCombinationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    limitations = list(STANDARD_LIMITATIONS)
    if efficacy_result["confidence"] < 0.5:
        limitations.append(
            f"Low bioactivity evidence for one or more of {request.compound_names} "
            f"(confidence={efficacy_result['confidence']}) — treat this prediction with extra caution."
        )
    if efficacy_result["discouraged_combination"]:
        limitations.append(efficacy_result["discouraged_reason"])

    return PredictionResponse(
        compound_names=request.compound_names,
        disease_parameter_deltas=efficacy_result["deltas"],
        side_effect_probabilities=se_result["side_effect_probabilities"],
        discouraged_combination=efficacy_result["discouraged_combination"],
        discouraged_reason=efficacy_result["discouraged_reason"],
        renal_adjustment_factor=efficacy_result["renal_adjustment_factor"],
        limitations=limitations,
    )


def _sse_event(event: str, data: dict) -> str:
    # jsonable_encoder handles the nested Pydantic models living inside
    # AgentState (CompoundQuery, ParsedPatient, DiseaseParameterDelta,
    # enums like Sex/DrugClass/Comorbidity) — plain json.dumps() would
    # choke on these directly.
    payload = json.dumps(jsonable_encoder(data))
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/simulate/stream")
async def simulate_stream(request: SimulateRequest):
    """Full agent pipeline (parse_query -> resolve_entities -> retrieve_data
    -> structure_features -> run_prediction -> retrieve_comparators ->
    generate_report) as SSE, one `event: <node_name>` per real node
    completion, plus a final `event: done` carrying the full report. See
    module docstring for why this exists instead of a plain POST."""

    async def event_generator():
        graph = build_graph()
        initial_state = {
            "raw_query": request.raw_query,
            "limitations": [],
            "parsed_patient": ParsedPatient(patient=request.patient, defaulted_fields=[]),
        }
        try:
            final_state: dict = {}
            async for chunk in graph.astream(initial_state):
                for node_name, state_update in chunk.items():
                    final_state.update(state_update)
                    yield _sse_event(node_name, state_update)
            yield _sse_event("done", final_state)
        except Exception as e:
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
