"""
parse_query node: raw NL -> ParseResult (compounds + patient + intent).

LOCAL MODEL BACKEND (see DECISIONS.md #7 for the Anthropic -> local pivot
rationale). Always routes through the QLoRA-fine-tuned model, served via
DIRECT IN-PROCESS INFERENCE (app/agent/local_finetuned_model.py), NOT
Ollama/GGUF — converting the merged model to GGUF for Ollama crashed the
host machine three times via OOM during the export's write phase, see
finetuning/session_log/actions.log for the full history. Loading the
already-merged model directly for inference (read-heavy, not write-heavy)
was verified safe in isolated testing. This is the ONE node actually
fine-tuned in this project.

REVISED (frontend build, see ui_build_brief.md cross-check): this node
used to branch — full extraction via the fine-tuned model when no patient
was pre-supplied, but a DIFFERENT, non-fine-tuned base-model/Ollama path
for compounds+intent-only extraction when a patient WAS pre-supplied
(e.g. Streamlit's structured form). That branch silently meant the
fine-tuned model was never exercised in the one flow every UI actually
uses (a free-text box + a separate structured patient form) — a real
mismatch between what DECISIONS.md #7 documents as "the one fine-tuned
node" and what a patient-form-using caller would actually get, and it
also meant this stage ran in a few fast seconds instead of the documented
~35-40s, silently invalidating the honest-pipeline-timing design the
frontend's animation is built around. Fixed: always call the fine-tuned
model for the compounds+intent extraction; when a real patient profile
is already supplied, its (usually-plausible but unverified) patient-guess
portion is simply discarded in favor of the caller-supplied one, rather
than switching to a different, unverified model/prompt path to avoid
that generation. Parses the raw response text with json.loads() +
model_validate(), NOT LangChain's with_structured_output()/tool-calling
path — that's the Anthropic-specific mechanism this pivot replaced, and
it doesn't match what the fine-tuned model was trained to produce (see
train_qlora.py's run_eval_pass, which uses this exact same parse-then-
validate approach during training evaluation — serving must match how it
was evaluated, not diverge from it).
"""

from app.agent.local_finetuned_model import FinetunedParseError, parse_with_finetuned_model
from app.agent.state import AgentState


class ParseQueryOutputError(Exception):
    """Raised when the local model's output isn't valid JSON or doesn't
    validate against the expected schema — surfaced explicitly rather
    than silently producing a half-populated state, since a parse
    failure here means nothing downstream can be trusted."""


async def parse_query(state: AgentState) -> AgentState:
    raw_query = state["raw_query"]
    limitations = state.get("limitations", [])
    patient_pre_supplied = state.get("parsed_patient") is not None

    # Takes ~35-40s per call (model load + generate, with an internal
    # retry on the known ~20% malformed-JSON failure mode — see
    # local_finetuned_model.py; VRAM is freed after each call rather than
    # kept resident) — acceptable for a backend pipeline step, same
    # tradeoff already accepted for generate_report's ~90-130s. This is
    # real work happening, not a delay to hide: the pipeline UI should
    # show it as such.
    #
    # If it fails after the internal retry, degrade to the same
    # clarification path resolve_entities uses for an unresolved compound
    # name, rather than let an uncaught exception crash the whole agent
    # run — a genuine model failure and a genuine "couldn't understand
    # the input" are both "ask the user to try again", not a 500 error.
    try:
        result, token_usage = await parse_with_finetuned_model(raw_query)
    except FinetunedParseError as e:
        limitations.append(
            "Could not parse your query into a structured request after retrying — please try "
            "rephrasing it. (Internal detail: " + str(e)[:200] + ")"
        )
        state["compounds"] = []
        state["all_resolved"] = False
        state["limitations"] = limitations
        if e.token_usage is not None:
            usage = dict(state.get("token_usage", {}))
            usage["parse_query"] = e.token_usage
            state["token_usage"] = usage
        return state

    usage = dict(state.get("token_usage", {}))
    usage["parse_query"] = token_usage
    state["token_usage"] = usage

    state["compounds"] = result.compounds
    state["question_intent"] = result.question_intent

    if patient_pre_supplied:
        # Caller-supplied patient data (typed into a form) is more
        # reliable than anything the model guessed from prose — keep
        # state["parsed_patient"] exactly as the caller set it, discard
        # result.parsed_patient entirely. Nothing to disclose as
        # "defaulted" here; the form values are real, not assumptions.
        state["limitations"] = limitations
        return state

    state["parsed_patient"] = result.parsed_patient
    if result.parsed_patient.defaulted_fields:
        limitations.append(
            "Patient fields not stated in your description were filled with clinical defaults: "
            f"{', '.join(result.parsed_patient.defaulted_fields)}. These are assumptions, not "
            "patient-confirmed values."
        )
    state["limitations"] = limitations
    return state
