"""
Generates synthetic (natural_language_query -> structured_output) pairs
for fine-tuning parse_query.

STEP 0 RECONCILIATION (2026-08-04) — this file originally imported from
reconstructed placeholder schemas (app/schemas/reference.py,
app/core/reference_registry.py, kept for reference under
finetuning/reference_schemas/) written by a prior session with no access
to the real repo. Reconciled against the REAL current schemas
(app/schemas/compound.py, app/agent/schemas.py, app/schemas/patient.py)
and app/core/drug_registry.py. Two real, non-cosmetic differences found:

1. CompoundQuery: the reference version matched already — parse_query's
   job is ONLY raw_name + dose_mg (resolved_name/drug_class/chembl_id/
   pubchem_cid/notes are resolve_entities' job, populated later, never
   by this node). No change needed there.

2. Patient shape: the reference generator produced a FLAT, NULLABLE
   patient dict ("omit fields the text doesn't state"). The real schema
   (ParsedPatient -> PatientProfile, see app/agent/schemas.py) requires
   EVERY field populated (nested under `baseline: DiseaseParameters`,
   no Optional/nullable fields) with a SEPARATE `defaulted_fields` list
   tracking which ones were guessed — this is the real, already-built,
   already-documented app design (see ParsedPatient's docstring and the
   real app/agent/parse_query.py's own SYSTEM_PROMPT), not a stylistic
   choice this pivot gets to relitigate. The generator below now builds
   that exact nested shape and threads through which fields were
   defaulted vs. actually rendered into the query text. It also adds
   baseline.egfr / baseline.serum_potassium, which don't exist in the
   reference schema at all and are NEVER mentioned in the rendered text
   templates — these are always-defaulted fields by construction, which
   is realistic (nobody casually states eGFR in a one-line query).

Target JSON / training system prompt come from
app/agent/local_llm_prompts.py (PARSE_QUERY_FULL_SYSTEM_PROMPT) rather
than being defined inline here — deliberately living in app/agent/ (the
deployed app), not here, since finetuning/ is one-off training tooling
that should depend on the app's real prompt, not the other way around.
See that module's docstring for why sharing one prompt (not copy-pasting
text into both the data generator and the serving code in
app/agent/parse_query.py) is the actual fix for train/serve prompt drift,
not just a comment asking someone to remember.

IMPORTANT DESIGN DECISION (unchanged from the original draft, still
correct): parse_query must NOT try to correct/resolve compound names —
that's resolve_entities' job (deterministic, ChEMBL-cache-backed fuzzy
matching). If you fine-tune the model to "helpfully" correct "loratan" to
"losartan" itself, you've duplicated resolve_entities' job inside the
LLM, made it unverifiable/unauditable (a hallucinated "correction" looks
identical to a real one), and broken the architecture's separation of
concerns. Typos are kept VERBATIM in the target output to train against
this failure mode, not just "for realism".

Run: python generate_synthetic_data.py --n 1500 --out data/
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent.local_llm_prompts import PARSE_QUERY_FULL_SYSTEM_PROMPT  # noqa: E402
from app.core.drug_registry import (  # noqa: E402 — path insert must come first
    DISCOURAGED_COMBINATIONS,
    HYPERTENSION_DRUGS,
    STANDARD_COMBINATIONS,
)

random.seed(3407)  # matches the Unsloth-convention seed used later, for traceable reproducibility

# Must match the defaults stated in local_llm_prompts.py's PARSE_QUERY_FULL_SYSTEM_PROMPT
# exactly — these are two views of the same fact (one prose, one code),
# not independent sources of truth. If you change one, change both.
DEFAULTS = {
    "age": 55,
    "sex": "male",
    "weight_kg": 80,
    "baseline.systolic_bp": 150,
    "baseline.diastolic_bp": 95,
    "baseline.heart_rate": 78,
    "baseline.egfr": 90,
    "baseline.serum_potassium": 4.2,
}

COMORBIDITY_PHRASES = {
    "type2_diabetes": ["type 2 diabetes", "T2DM", "diabetic"],
    "chronic_kidney_disease": ["chronic kidney disease", "CKD", "reduced kidney function"],
    "heart_failure": ["heart failure", "HF", "a history of heart failure"],
    "asthma_copd": ["asthma", "COPD"],
}

QUESTION_INTENT_PHRASES = {
    "efficacy": ["How effective would this be", "What blood pressure reduction can I expect", "Will this control their BP"],
    "side_effects": ["What side effects should I watch for", "Any safety concerns", "What are the risks"],
    "compare_to_standard": ["How does this compare to standard first-line therapy", "Is this better than the usual combo"],
    "general": ["What would happen if I gave this", "Can you assess this regimen", "Talk me through this case"],
}

PATIENT_TEMPLATES = [
    "a {age}-year-old {sex}{weight_clause}, BP {sys}/{dia}{hr_clause}{comorb_clause}",
    "patient: {age}yo {sex}{weight_clause}, presenting with BP of {sys}/{dia}{hr_clause}{comorb_clause}",
    "{age} year old {sex_full}, current vitals {sys}/{dia} mmHg{hr_clause}{comorb_clause}",
    "a {sex_full} in their {decade}s{weight_clause}, BP reading {sys} over {dia}{comorb_clause}",
]

COMPOUND_TEMPLATES = [
    "I'm considering {compounds} for",
    "What if I prescribed {compounds} to",
    "Evaluating {compounds} for",
    "{compounds} - would this work for",
]


def maybe_typo(name: str, rate: float = 0.15) -> str:
    """Inject a realistic typo at the given rate. Returns name unchanged
    otherwise. This is what teaches the model to extract verbatim rather
    than 'auto-correct' - the target output must use this SAME typo'd
    string, not the corrected one."""
    if random.random() > rate or len(name) < 4:
        return name
    i = random.randint(1, len(name) - 2)
    kind = random.choice(["swap", "drop", "double"])
    if kind == "swap":
        chars = list(name)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    elif kind == "drop":
        return name[:i] + name[i + 1:]
    else:
        return name[:i] + name[i] + name[i:]


def sample_patient() -> dict:
    age = random.randint(35, 85)
    sex = random.choice(["male", "female"])
    weight = round(random.uniform(50, 110), 1)
    diastolic = round(random.uniform(70, 105))
    systolic = round(diastolic + random.uniform(25, 65))  # keeps systolic > diastolic, matches the real validator
    hr = round(random.uniform(55, 100))

    n_comorb = random.choices([0, 1, 2], weights=[0.5, 0.35, 0.15])[0]
    comorbs = random.sample(list(COMORBIDITY_PHRASES.keys()), k=n_comorb)

    # ~20% of examples omit weight/heart-rate from the rendered text -
    # trains the model to default+disclose rather than hallucinate a
    # stated value when the clinician's text genuinely doesn't mention
    # it. age/sex/systolic/diastolic are never omitted (every template
    # states them) - that's a realistic assumption (a clinical query
    # about a patient essentially always states who they are and their
    # BP), not a gap in defaulting coverage. egfr/serum_potassium are
    # NEVER in the text at all (see render_patient_text) - always
    # defaulted, unconditionally, below.
    omit_weight = random.random() < 0.2
    omit_hr = random.random() < 0.2

    return {
        "age": age, "sex": sex, "weight_kg": weight,
        "systolic_bp": systolic, "diastolic_bp": diastolic,
        "heart_rate": hr, "comorbidities": comorbs,
        "_omit_weight": omit_weight, "_omit_hr": omit_hr,
    }


def render_patient_text(p: dict) -> str:
    template = random.choice(PATIENT_TEMPLATES)

    comorb_clause = ""
    if p["comorbidities"]:
        phrases = [random.choice(COMORBIDITY_PHRASES[c]) for c in p["comorbidities"]]
        comorb_clause = f", with {' and '.join(phrases)}"

    # Build weight/HR clauses compositionally - empty string when omitted,
    # rather than substituting a placeholder into the template and trying
    # to strip it back out afterward (that approach silently left "?"
    # artifacts in ~14% of examples across templates that didn't match
    # the exact cleanup string - caught by testing the actual output,
    # not by re-reading the code).
    if p["_omit_weight"]:
        weight_clause = ""
    else:
        weight_clause = random.choice([
            f" weighing {p['weight_kg']}kg", f", {p['weight_kg']}kg", f", roughly {p['weight_kg']}kg",
        ])

    if p["_omit_hr"]:
        hr_clause = ""
    else:
        hr_clause = random.choice([
            f", heart rate {p['heart_rate']}", f" and HR {p['heart_rate']}", f", pulse {p['heart_rate']}",
        ])

    return template.format(
        age=p["age"], sex=p["sex"][0].upper(), sex_full=p["sex"],
        weight_clause=weight_clause, sys=p["systolic_bp"], dia=p["diastolic_bp"],
        hr_clause=hr_clause, comorb_clause=comorb_clause, decade=p["age"] // 10 * 10,
    )


def build_target_patient(p: dict) -> tuple[dict, list[str]]:
    """Builds the full, always-populated `parsed_patient` structure the
    real PatientProfile schema requires, and the list of which fields
    were defaulted rather than read from the rendered text. egfr and
    serum_potassium are unconditionally defaulted — the text templates
    never mention them (see render_patient_text)."""
    defaulted_fields = ["baseline.egfr", "baseline.serum_potassium"]

    weight_kg = DEFAULTS["weight_kg"] if p["_omit_weight"] else p["weight_kg"]
    if p["_omit_weight"]:
        defaulted_fields.append("weight_kg")

    heart_rate = DEFAULTS["baseline.heart_rate"] if p["_omit_hr"] else p["heart_rate"]
    if p["_omit_hr"]:
        defaulted_fields.append("baseline.heart_rate")

    comorbidities = p["comorbidities"] if p["comorbidities"] else ["none"]

    patient = {
        "age": p["age"],
        "sex": p["sex"],
        "weight_kg": weight_kg,
        "baseline": {
            "systolic_bp": p["systolic_bp"],
            "diastolic_bp": p["diastolic_bp"],
            "heart_rate": heart_rate,
            "egfr": DEFAULTS["baseline.egfr"],
            "serum_potassium": DEFAULTS["baseline.serum_potassium"],
        },
        "comorbidities": comorbidities,
        "current_medications": [],
    }
    return patient, defaulted_fields


def sample_compounds() -> list[dict]:
    """Weighted sampling across single drug / standard combo / discouraged
    combo / thiazide-involving combo, so the fine-tuned model sees all
    the cases it'll actually be asked about downstream."""
    mode = random.choices(
        ["single", "standard_combo", "discouraged_combo", "thiazide_combo"],
        weights=[0.4, 0.3, 0.1, 0.2],
    )[0]

    by_class = {}
    for d in HYPERTENSION_DRUGS:
        by_class.setdefault(d["drug_class"], []).append(d["name"])

    if mode == "single":
        drug = random.choice(HYPERTENSION_DRUGS)["name"]
        chosen = [drug]
    elif mode == "discouraged_combo":
        c1, c2 = random.choice(DISCOURAGED_COMBINATIONS)
        chosen = [random.choice(by_class[c1]), random.choice(by_class[c2])]
    elif mode == "thiazide_combo":
        from app.schemas.compound import DrugClass

        other_classes = [c for c in by_class if c != DrugClass.THIAZIDE_DIURETIC]
        c2 = random.choice(other_classes)
        chosen = [random.choice(by_class[DrugClass.THIAZIDE_DIURETIC]), random.choice(by_class[c2])]
    else:
        c1, c2 = random.choice(STANDARD_COMBINATIONS)
        chosen = [random.choice(by_class[c1]), random.choice(by_class[c2])]

    result = []
    for name in chosen:
        raw = maybe_typo(name)
        dose = random.choice([None, None, None, round(random.choice([2.5, 5, 10, 20, 25, 40, 50]))])
        result.append({"raw_name": raw, "dose_mg": dose})
    return result


def render_compound_text(compounds: list[dict]) -> str:
    parts = []
    for c in compounds:
        if c["dose_mg"]:
            parts.append(f"{c['raw_name']} {c['dose_mg']}mg")
        else:
            parts.append(c["raw_name"])
    joined = " and ".join(parts) if len(parts) == 2 else ", ".join(parts)
    template = random.choice(COMPOUND_TEMPLATES)
    return template.format(compounds=joined)


def generate_example() -> dict:
    patient = sample_patient()
    compounds = sample_compounds()
    intent = random.choice(list(QUESTION_INTENT_PHRASES.keys()))
    intent_phrase = random.choice(QUESTION_INTENT_PHRASES[intent])

    compound_text = render_compound_text(compounds)
    patient_text = render_patient_text(patient)
    query = f"{intent_phrase}? {compound_text} {patient_text}."

    target_patient, defaulted_fields = build_target_patient(patient)

    target = {
        "compounds": [{"raw_name": c["raw_name"], "dose_mg": c["dose_mg"]} for c in compounds],
        "parsed_patient": {
            "patient": target_patient,
            "defaulted_fields": defaulted_fields,
        },
        "question_intent": intent,
    }

    return {
        "messages": [
            {"role": "system", "content": PARSE_QUERY_FULL_SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = [generate_example() for _ in range(args.n)]
    random.shuffle(examples)
    n_val = int(args.n * args.val_fraction)
    val, train = examples[:n_val], examples[n_val:]

    with open(out_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open(out_dir / "val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(train)} train examples, {len(val)} val examples to {out_dir}/")
    print("\n--- Sample example ---")
    sample = examples[0]
    print("USER:", sample["messages"][1]["content"])
    print("ASSISTANT:", sample["messages"][2]["content"])


if __name__ == "__main__":
    main()
