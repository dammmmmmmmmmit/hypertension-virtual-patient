"""
generate_report node: synthesizes everything gathered by the earlier
nodes into the clinician-readable report. Every report must have the 5
sections from the continuation brief Section 10: (1) compound(s) +
grounded mechanism, (2) predicted efficacy, (3) predicted side effects
with patient-adjusted risk, (4) comparison vs standard drugs/combos, (5)
explicit caveats — never skip the caveats section, regardless of how
confident the numbers look (Critical Engineering Decision #6).

If entity resolution or prediction failed (ambiguous name, unsupported
combination size, etc.), this returns a short, DETERMINISTIC clarification
message built directly from the recorded notes/limitations — no LLM call.
There's no synthesis to do when the honest answer is "I need you to
confirm something first," and skipping the LLM call here removes a
hallucination risk for exactly the situation where accuracy matters most
(the user's input wasn't even understood correctly yet).

LLM backend: local Ollama (qwen2.5:14b-instruct by default), not the
Anthropic API — see DECISIONS.md #7 for the pivot rationale. This node is
inference-only and NEVER fine-tuned (unlike parse_query): open-ended
report prose has no gold-standard dataset to fine-tune against, and
fabricating one would be worse than not fine-tuning at all. A capable
off-the-shelf instruct model with good prompting is the right tool here.
"""

from langchain_ollama import ChatOllama

from app.agent.state import AgentState
from app.core.settings import settings

SYSTEM_PROMPT = """You write clinician-readable reports for a hypertension drug-response \
screening tool. You are given: resolved compound(s) with mechanism-of-action context retrieved \
from a database, ML model predictions (efficacy deltas, side-effect probabilities), comparator \
standard-of-care combinations, and a list of limitations that MUST all appear in your caveats \
section, verbatim or near-verbatim — do not soften or drop any of them.

Structure your report with exactly these 5 sections, in this order:
1. Compound(s) identified and mechanism summary — ground this ONLY in the mechanism_text \
provided in the context below. Do not add pharmacological claims beyond what's given.
2. Predicted efficacy — state the predicted systolic/diastolic BP change and what it implies \
for the stated baseline. Mention the confidence level in plain language, not just the number. \
If a renal adjustment factor other than 1.0 was applied, say so HERE (efficacy section) — it is \
a multiplier on the BP prediction only, not on the side-effect probabilities.
3. Predicted side-effect profile — list the top predicted side effects in ranked order (most to \
least likely). The underlying model's raw probability outputs are near-saturated — consistently \
~0.99+ regardless of which drug, an artifact of training on only 14 rows — and are NOT calibrated. \
Do NOT state them as percentages or imply precise likelihoods (e.g. do not write "probability = \
1.00"); describe them qualitatively by rank instead (e.g. "most prominently predicted: X, Y" / \
"also predicted, though lower-ranked: Z"). Say explicitly that ranking is meaningful here but the \
absolute probability value is not. Do NOT attribute these to the patient's renal adjustment factor \
— the side-effect model does not use renal function as an input; only the efficacy prediction is \
renal-adjusted.
4. Comparison vs standard-of-care — briefly compare against the comparator combination(s) given.
5. Caveats — a bulleted list including every item in the limitations list given below, plus \
always: "This is a screening/demo tool, not clinical guidance."

If a "DISCOURAGED COMBINATION WARNING" line is present in the context below, it must appear \
prominently near the TOP of the report (before section 1), not buried in the caveats — dual \
RAAS blockade and similar flagged combinations are a safety signal, not a footnote.

If NO "DISCOURAGED COMBINATION WARNING" line is present in the context (e.g. a single-compound \
report, or a combination not flagged), do NOT mention dual RAAS blockade, discouraged \
combinations, or any other combination-risk warning anywhere in the report — inventing a safety \
warning that wasn't actually flagged is worse than omitting one, since it's exactly the kind of \
plausible-sounding but ungrounded claim this tool exists to avoid making.

Keep the tone factual and clinical, not promotional. Do not invent numbers, mechanisms, or \
warnings not given to you in the context below — if it's not in the context, it doesn't go in \
the report."""


def _build_llm():
    # temperature=0.2-0.8ish is reasonable for report prose (contrast with
    # parse_query's temperature=0.1 — structured extraction wants
    # consistency, report writing benefits from not reading robotic).
    return ChatOllama(
        model=settings.generate_report_model,
        base_url=settings.ollama_base_url,
        temperature=0.4,
    )


def _clarification_report(state: AgentState) -> str:
    lines = ["**Could not complete this simulation — need clarification:**\n"]
    for compound in state.get("compounds", []):
        if compound.notes:
            lines.append(f"- **{compound.raw_name}**: {compound.notes}")
    for limitation in state.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.append("\nThis is a screening/demo tool, not clinical guidance.")
    return "\n".join(lines)


def _format_context(state: AgentState) -> str:
    parts = []

    compounds_desc = ", ".join(
        f"{c.resolved_name} ({c.drug_class.value if c.drug_class else 'unknown'})" for c in state["compounds"]
    )
    parts.append(f"Resolved compounds: {compounds_desc}")

    for ctx in state.get("mechanism_contexts", []):
        tag = "ChEMBL-recorded" if ctx["source"] == "chembl_mechanism" else "class-level inference, not compound-specific"
        parts.append(f"Mechanism ({ctx['drug_name']}, {tag}): {ctx['mechanism_text']}")

    prediction = state.get("prediction")
    if prediction:
        for delta in prediction["deltas"]:
            parts.append(
                f"Predicted {delta.parameter}: baseline {delta.baseline_value} -> "
                f"{delta.predicted_value:.1f} (delta {delta.predicted_delta:+.1f}), "
                f"confidence={delta.confidence}"
            )
        parts.append(f"Renal adjustment factor applied: {prediction['renal_adjustment_factor']}")
        top_se = list(prediction["side_effect_probabilities"].items())[:8]
        # Deliberately NOT passing the raw probability values here (e.g.
        # "Cough (1.00)") — they're near-saturated (~0.99+) regardless of
        # drug, an n=14-training-row artifact, not a calibrated
        # likelihood. Feeding the LLM a literal 2dp number invites it to
        # assert false-precision claims like "probability = 1.00" for 8
        # different side effects at once. Only rank order is meaningful
        # here (see DECISIONS.md #4) — that's what the model gets.
        parts.append(
            "Top predicted side effects, in ranked order (most to least likely; do not state "
            "these as percentages — see system instructions): " + ", ".join(name for name, _ in top_se)
        )

    if state.get("discouraged_warning"):
        parts.append(f"DISCOURAGED COMBINATION WARNING: {state['discouraged_warning']}")

    for comp in state.get("comparators", []):
        parts.append(f"Comparator: {comp['description']}")

    parts.append("Limitations to include in caveats: " + "; ".join(state.get("limitations", [])))

    return "\n".join(parts)


async def generate_report(state: AgentState) -> AgentState:
    if not state.get("all_resolved", False) or state.get("prediction") is None:
        state["report"] = _clarification_report(state)
        return state

    llm = _build_llm()
    context = _format_context(state)
    response = await llm.ainvoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", context),
        ]
    )
    state["report"] = response.content

    # response.usage_metadata is populated by langchain-ollama from
    # Ollama's own prompt_eval_count/eval_count fields — verified live
    # against the running model, not assumed from the LangChain docs.
    usage_meta = response.usage_metadata or {}
    token_usage = dict(state.get("token_usage", {}))
    token_usage["generate_report"] = {
        "model": response.response_metadata.get("model", settings.generate_report_model),
        "input_tokens": usage_meta.get("input_tokens", 0),
        "output_tokens": usage_meta.get("output_tokens", 0),
        "total_tokens": usage_meta.get("total_tokens", 0),
        "attempts": 1,
    }
    state["token_usage"] = token_usage
    return state
