"""
System prompt for parse_query's local model backend (see DECISIONS.md #7
for the Anthropic -> local-model pivot). Living here, in app/agent/ (the
deployed app), not under finetuning/, is deliberate: finetuning/ is
one-off training tooling that depends on the app's real schemas, not the
other way around — the app must not have a runtime dependency on a
training-scripts directory. finetuning/generate_synthetic_data.py imports
PARSE_QUERY_FULL_SYSTEM_PROMPT from here, precisely so the training data
and the serving code can never independently drift (the same risk
train_qlora.py's formatting_func and Modelfile.parse_query's chat-
template comment both flag for their own layers).

PARSE_QUERY_FULL_SYSTEM_PROMPT is used for EVERY parse_query call, always
via the fine-tuned model (app/agent/local_finetuned_model.py) — there is
no separate base-model path anymore. A lighter compounds-only prompt used
to exist for the case where a patient profile was already supplied (e.g.
a structured form), served by a different, non-fine-tuned model; that
silently meant the fine-tuned model was never exercised in the one flow
every real UI actually uses, and that stage ran in a few fast seconds
instead of its documented ~35-40s. Removed — parse_query.py now always
calls the fine-tuned model and simply discards its patient-guess when a
real one is already available. See DECISIONS.md #7 and parse_query.py.

Written for RAW JSON TEXT output (not LangChain `with_structured_output`
tool-calling/schema-binding) because that's what the QLoRA-fine-tuned
model was actually trained to produce.
"""

PARSE_QUERY_FULL_SYSTEM_PROMPT = """You extract structured data from a clinician's natural-language \
description of candidate hypertension drug(s) and a patient, for a screening simulation tool.

Extract ONLY what is literally stated for compounds - do not correct spelling or resolve a \
compound name to its "real" name (that is a separate downstream step); raw_name must be copied \
EXACTLY as written, typos included.

The patient profile schema requires EVERY field below to be populated (no nulls) - if the \
text does not state a field, fill it with the standard default listed and add that field's \
name to defaulted_fields. Do not guess a non-default value for an unstated field.

Defaults to use when not stated in the text:
- age: 55, sex: "male", weight_kg: 80
- baseline.systolic_bp: 150, baseline.diastolic_bp: 95, baseline.heart_rate: 78
- baseline.egfr: 90, baseline.serum_potassium: 4.2
- comorbidities: ["none"]

Respond with ONLY a JSON object matching this exact shape, no other text:
{
  "compounds": [{"raw_name": "<exactly as written>", "dose_mg": <number or null>}],
  "parsed_patient": {
    "patient": {
      "age": <int>, "sex": "<male|female>", "weight_kg": <number>,
      "baseline": {
        "systolic_bp": <number>, "diastolic_bp": <number>, "heart_rate": <number>,
        "egfr": <number>, "serum_potassium": <number>
      },
      "comorbidities": [<list of strings from: none, type2_diabetes, chronic_kidney_disease, \
heart_failure, asthma_copd>],
      "current_medications": []
    },
    "defaulted_fields": [<every field name you defaulted rather than read from the text, \
e.g. "weight_kg", "baseline.heart_rate", "baseline.egfr">]
  },
  "question_intent": "<efficacy|side_effects|compare_to_standard|general>"
}"""
