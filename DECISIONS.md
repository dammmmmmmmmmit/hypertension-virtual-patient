# Design Decisions Log

This file records decisions that affect scientific/statistical validity of the
system, so they can survive a viva/defense question without having to be
re-derived from scratch. Add to it; don't silently change approach elsewhere
without updating this file.

---

## 1. Efficacy label source: class-level meta-analysis anchor + disclosed
   potency-based adjustment (resolves Section 8 of the continuation brief)

### The problem

ChEMBL bioactivity (`pchembl_value` / manually-derived `pX`, see
`potency_utils.py`) is a molecular binding-affinity measure: -log10 of the
concentration needed for 50% target engagement in an isolated assay. The
thing we actually want to predict — mmHg reduction in systolic/diastolic BP
in a person — is a whole-organism pharmacodynamic outcome shaped by potency,
but also by dose, bioavailability, protein binding, half-life, and
downstream physiology that potency alone doesn't capture. Wiring
`mean_pchembl` directly into `DiseaseParameterDelta.predicted_value` as if
it were an mmHg value would be dimensionally meaningless and scientifically
indefensible — a losartan pX of 8.52 and an amlodipine pX of 5.56 do NOT
mean losartan lowers BP ~1000x more; they're different targets, different
assay types, not on a comparable absolute scale even within a class.

### Verified primary-source anchor

Re-derived from the actual papers (not from memory) via live lookup on
2026-07-28:

**Law MR, Morris JK, Wald NJ. "Use of blood pressure lowering drugs in the
prevention of cardiovascular disease: meta-analysis of 147 randomised
trials in the context of expectations from prospective epidemiological
studies." BMJ 2009;338:b1665.** Mean reduction at standard dose, by class
(cross-checked against the NIHR HTA executive summary of the same
programme of work, NCBI Bookshelf NBK62259, which reproduces the table):

| Class | Systolic Δ (mmHg) | Diastolic Δ (mmHg) |
|---|---|---|
| Thiazide diuretics | 8.8 | 4.4 |
| Beta-blockers | 9.2 | 6.7 |
| ACE inhibitors | 8.5 | 4.7 |
| ARBs | 10.3 | 5.7 |
| Calcium-channel blockers | 8.8 | 5.9 |

**Combination additivity** (same source): combining two classes is
approximately additive but not perfectly so. The paper's worked example
from a starting diastolic pressure of 90 mmHg: 1 drug → 4.7 mmHg reduction,
2 drugs → 8.9 mmHg (94.7% of naive sum 9.4), 3 drugs → 12.6 mmHg (89.4% of
naive sum 14.1). This is a mild, roughly-linear-in-log sub-additivity
discount — each added drug contributes at ~95% of its full-monotherapy
effect relative to the previous step.

Corroborating figure from a companion paper — **Wald DS, Law M, Morris JK,
Bestwick JP, Wald NJ. "Combination therapy versus monotherapy in reducing
blood pressure: meta-analysis on 11,000 participants from 42 trials." Am J
Med. 2009;122(3):290-300** — thiazide monotherapy: 7.3 mmHg systolic
placebo-subtracted; thiazide + one other class: 14.6 mmHg (~2x, i.e. closer
to fully additive for this specific thiazide-anchored combination set than
the general 2-class table above). We treat the general Law/Morris/Wald
table's sub-additivity discount as the default (it's the more granular,
matched-methodology source across 1/2/3 drugs), and note the AJM figure as
consistent-direction corroboration, not a contradiction — different drug
subsets, similar conclusion ("additive, if anything mildly sub-additive").

### Decision

Class-level trial data is the **efficacy label anchor** (ground truth for
"what a standard-dose drug in this class does to BP on average"), not
per-compound ChEMBL potency. ChEMBL potency, RDKit descriptors, and patient
covariates are **features** that explain plausible *within-class*
deviation from that anchor — not the label itself.

Concretely, per-drug training label construction:

```
baseline_delta(class)          = class table value above (systolic & diastolic)
potency_z(drug)                 = (drug's pX - class mean pX) / class std pX
                                   (NaN potency, i.e. the 3 thiazides -> z = 0,
                                   i.e. "assume class-average", not imputed as
                                   a fabricated number — consistent with
                                   Critical Engineering Decision #4)
adjusted_delta(drug)             = baseline_delta(class) * (1 + ALPHA * clip(potency_z, -2, 2))
                                   ALPHA = 0.08 (chosen small deliberately —
                                   see "honesty" note below)
```

For combinations, apply the additivity discount empirically observed above
(~95% marginal contribution per added drug beyond the first) to the sum of
each component's `adjusted_delta`, then apply
`PatientProfile.renal_adjustment_factor()` (and any other covariate
multiplier) as a final, explicit, disclosed multiplier on top — never
baked invisibly into the model itself (Critical Engineering Decision #5).

### Why ALPHA = 0.08 and not fit from data — the honesty caveat

There is no public per-compound (as opposed to per-class) RCT dataset
covering all 15 registry drugs with consistent dosing/methodology, so there
is no real ground truth to fit a within-class adjustment coefficient
against. **ALPHA is a deliberately small, hand-set, documented heuristic
bound, not a value learned from or validated against real per-compound
clinical outcomes.** This is disclosed here and must be disclosed in every
generated report and in the README limitations section — do not let this
caveat erode over the remaining sessions.

Given this, the LightGBM efficacy model is trained on a **label set that is
itself semi-synthetic** (real class-level anchor + a small disclosed
heuristic perturbation), not on raw clinical trial outcomes. What real
signal *is* there for the model to learn, and why train an ML model at all
rather than just hand-coding the formula above?

1. **Combination interaction learning** — the additivity-discount and
   `DISCOURAGED_COMBINATIONS` flagging are genuinely non-trivial functions
   of which two classes are combined; letting the model learn this from
   engineered interaction features (rather than every combination rule
   being hand-written) is a legitimate small ML task and demonstrates the
   feature-engineering/model pipeline the project needs to show.
2. **Patient covariate interaction** — renal function, comorbidities, and
   drug class don't combine in a strictly linear way clinically (e.g.
   beta-blocker caution in asthma/COPD isn't a BP-magnitude effect at all,
   it's a distinct side-effect-risk channel); the multi-task setup lets
   these show up in the side-effect head (task 2), which HAS a real,
   non-synthetic label source (SIDER), while task 1 stays anchored to
   trial-level truth.
3. Task 2 (side-effect probability) and, to a lesser extent, task 3
   (interaction term, evaluated primarily on whether it reproduces the
   documented discount + flags discouraged combos correctly rather than
   novel unseen chemistry) carry the actual predictive/statistical
   learning weight of this project. Task 1 is best understood as a
   **calibrated, ML-smoothed lookup**, not a discovery model — say so
   explicitly in the report and the model card / README, not just here.

This is the single biggest scientific-honesty tradeoff in the project.
Do not present task 1's output with more confidence than "class-level
trial evidence, adjusted by a small heuristic factor for this specific
compound's relative potency within its class" — see Critical Engineering
Decisions #5, #6, #8 in the continuation brief, all of which this decision
must remain consistent with.

### Alternatives considered and rejected

- **Raw potency → mmHg regression.** Rejected: no valid unit conversion
  path exists from a target-binding affinity to a whole-organism outcome;
  would be presenting a category error as a trained model.
- **Per-compound literature mining for individual RCT dose-response
  figures for all 15 drugs.** Would be the most scientifically defensible
  option, but is a multi-week literature synthesis task on its own
  (heterogeneous trial designs, doses, populations) and is out of scope
  for a 2-week project. Flagged in the README as the clear "if I had more
  time" next step.
- **Skip an efficacy ML task entirely, output the class table directly.**
  Rejected: it's honest but throws away the project's core "equal
  engineering care across every layer" requirement (Section 2) for one of
  three prediction tasks; the semi-synthetic approach above at least
  demonstrates the pipeline while being explicit about its limits.

---

---

## 2. SIDER / STITCH compound ID mapping — verified, not assumed

The continuation brief flagged this as something to verify rather than trust
from memory: "STITCH flat compound IDs are PubChem CIDs offset by
+100,000,000." Verified 2026-07-28 against two independent sources (a
Biostars explanation of STITCH's `CID0`/`CID1` convention, and a public
`drug_id_mapping` cross-reference project showing real STITCH ID examples),
cross-checked against each other:

- SIDER's `meddra_all_se.tsv` / `meddra_freq.tsv` column 1 is the **flat**
  STITCH ID: chemical identity merged across stereoisomers/salts. Format is
  literal string `CID1` + PubChem CID zero-padded to 8 digits. As an
  integer (stripping the `CID` prefix, keeping the leading `1`), this
  equals `PubChem_CID + 100_000_000`. **This is the column to join against**
  — it's stereochemistry-agnostic, matching how we resolve one canonical
  PubChem CID per generic drug name.
- Column 2 is the **stereo** STITCH ID (`CID0` + zero-padded CID): as an
  integer this equals the PubChem CID directly (no offset). Not used here —
  we don't track which specific stereoisomer SIDER's label data refers to,
  and our registry resolves one CID per drug name via PubChem's own
  name-search (which returns the parent/canonical compound).

**Revised during implementation (2026-07-28):** the plan above — resolve
our own PubChem CID, compute `flat_stitch_id = cid + 100_000_000`, join —
broke on lisinopril. PubChem's name search resolves "lisinopril" to CID
5362119 (the specific (2S,2S,1S) stereoisomer, titled "Lisinopril" on
PubChem), but SIDER's own `drug_names.tsv` maps the name "Lisinopril" to
CID 3937 — the flat/achiral parent structure, same molecular formula, no
stereo descriptors. PubChem assigns many CIDs to what's clinically one
drug (salts, stereoisomers, protonation states); `cids_type=same_
connectivity` on 5362119 returns 20+ related CIDs with no reliable rule
for picking whichever one SIDER happened to use when it was built.

**Actual implementation: join on drug name against SIDER's own
`drug_names.tsv` (case-insensitive exact match), not on an independently
re-resolved PubChem CID.** SIDER's name file states exactly which STITCH
ID it used for a given generic name, sidestepping the CID-ambiguity
problem entirely. `pubchem_cid` is still resolved separately per drug (via
PubChem PUG-REST name search) and stored, but it is a descriptive field,
not the SIDER join key. See `app/ingestion/sider_client.py` —
`get_side_effects_for_drug(name)` is the primary path;
`get_side_effects_for_cid(cid)` is kept only as a documented fallback.

This is exactly the kind of thing Section 8 of the brief warned about:
verify from primary sources, don't assume a clean ID mapping holds for
every record — it held for 14/15 drugs by name, and the one exception
(enalapril) turned out to be a genuine SIDER coverage gap either way, not
an artifact of which ID we joined on.

---

---

## 3. Model task design: two ML tasks, not three — "combination interaction"
   is absorbed into task 1, not a separate classifier

Section 9 of the continuation brief lists three model tasks: (1) efficacy,
(2) side-effect probability, (3) "combination interaction term". Resolving
task 3 literally — a standalone classifier over class-pair combinations —
runs into a real problem: there are only 5 drug classes in this registry,
so there are only C(5,2) = 10 possible class-pairs, of which exactly 1
(ACE inhibitor + ARB) is labeled "discouraged". Training a classifier on
10 rows with 1 positive would be memorization dressed up as ML, not a
statistically meaningful model — the opposite of the honesty this project
is supposed to demonstrate.

**Decision:** fold the combination-interaction signal into task 1
(efficacy) instead of giving it a separate head:

- Task 1's training set includes BOTH single-drug rows and 2-drug
  combination rows (see DECISIONS.md #1 for the label formula, which
  already encodes the empirical sub-additivity discount for combos). The
  efficacy model therefore has to learn the difference between a
  single-drug response and a combined one directly from the data — that
  IS the combination-interaction task, just folded into task 1 rather than
  split out.
- `DISCOURAGED_COMBINATIONS` flagging stays a **hard-coded rule** read
  from `drug_registry.py`, not a model output. This is deliberate, not a
  scope cut: whether dual RAAS blockade is dangerous is a fixed clinical
  fact, not a statistical pattern with useful uncertainty — presenting it
  with a model confidence score would UNDERSTATE how well-established the
  risk is. Safety-critical flags should not be softer than the evidence
  behind them.

Net result: **two trained ML tasks** (efficacy regression, side-effect
multi-label classification) plus one **rule-based safety check**
(discouraged-combination flag), rather than three ML tasks where one would
have been theater. If asked at a viva why there are only two model heads
despite the original three-task plan, point here.

---

## 4. Side-effect label vocabulary must exclude near-universal terms —
   found by testing the deployed model, not by code review

Initial implementation of `build_side_effect_vocabulary()` picked the
top-N most frequent PT-level side effects across the registry. This
compiled, ran, and produced plausible-looking per-label LOO-CV accuracy
(~0.90) — but testing the actual FastAPI endpoint end-to-end (not just
unit-level checks) surfaced the real problem: **atenolol and
hydrochlorothiazide — a beta-blocker and a thiazide, pharmacologically
unrelated — returned byte-identical top-10 side-effect predictions, all
at ~0.9999999 probability.**

Root cause: SIDER's label-derived side-effect data comes from FDA drug
labels, which routinely list generic adverse events (headache, nausea,
dizziness, rash, constipation...) for nearly every drug regardless of
mechanism — this is a feature of how drug labels get written (broad
"adverse reactions" sections, low bar for inclusion), not a real
pharmacological signal. In this 14-drug registry, the naive top-30 by
frequency was dominated by terms present in **all 14/14** drugs. A
classifier trained on an all-1 column has nothing to discriminate — it
correctly learns "always predict 1," which is technically high-accuracy
and completely useless for a report meant to differentiate drugs.

**Fix:** filter the vocabulary to side effects with real variance across
the registry — present in at least 2 but no more than (n_drugs_with_data
- 2) of the drugs — before taking the top-N by frequency within that
band. This is why LOO-CV per-label accuracy after the fix is expected to
be lower than the pre-fix ~0.90 — that's the honest number; the earlier
one was inflated by unlearnable constant columns. See `dataset.py`
`build_side_effect_rows()`.

**Lesson for future sessions:** a model that trains without errors and
reports a plausible-looking CV metric is not the same as a model that
behaves sensibly on real inputs — always spot-check predictions across a
couple of genuinely different inputs before trusting a metric, especially
at this dataset size.

---

## 5. Losartan was resolved to the wrong compound — found while building
   the Week-2 RAG ingestion, not during Week 1 review

While fetching mechanism-of-action text from ChEMBL's `/mechanism.json`
for the Qdrant RAG index (Week 2), losartan came back with zero
mechanism records — worth checking directly rather than assuming it's
just another data gap like the thiazides. It wasn't: **the continuation
brief's own "verified fact" about losartan (Kb = 3.0 nM, pX ≈ 8.52,
recovered via the Critical Lesson #2 fallback) was based on the wrong
ChEMBL compound.**

`ChEMBLClient.search_molecule_by_name("losartan")` (the fuzzy full-text
`/molecule/search.json?q=losartan` endpoint) returns `CHEMBL382821`
("LOSARTAN NITROOXY ESTER") as its top-ranked hit — a distinct
NO-donating losartan derivative with its own SMILES
(`CCCCc1nc(Cl)c(CO[N+](=O)[O-])n1...` vs. real losartan's
`CCCCc1nc(Cl)c(CO)n1...`) and its own `molecule_hierarchy` (parent =
itself). The actual "LOSARTAN" entry is `CHEMBL191`
(`molecule.json?pref_name__iexact=losartan`), a completely separate
ChEMBL ID. Unlike every other salt-form case in this registry (e.g.
"AMLODIPINE BESYLATE" search hit -> `parent_chembl_id` correctly
resolves to plain "AMLODIPINE"), the parent-ID resolution step doesn't
rescue this one, because CHEMBL382821 isn't a salt of losartan at all —
it's a different molecule that ChEMBL's relevance ranking happened to
rank first for the query string "losartan". Checked all other 14
registry drugs for the same failure mode after finding this — none of
the others were affected (verified via `molecule.json?pref_name__iexact`
against each search hit's `pref_name`).

**Fix:** `search_molecule_by_name()` now tries an exact
`pref_name__iexact` match first, falling back to the fuzzy full-text
search only if no exact match exists (needed for real salt-form cases
like "enalapril" -> "ENALAPRIL MALEATE" as the top literal hit, where an
exact-name entry also happens to separately exist and get preferred
correctly). See `app/ingestion/chembl_client.py`.

**Effect of the fix:** losartan now resolves to `CHEMBL191`, with 30 real
bioactivity records and a directly ChEMBL-computed `mean_pchembl = 7.98`
— it no longer needs the single-record Kb fallback at all. The dataset,
trained models, and cached `resolved_compounds` row for losartan were all
rebuilt after this fix; every earlier number attributed to "losartan" in
this project prior to 2026-07-28 (including the brief's own worked
example) was actually describing losartan nitrooxy ester.

**Lesson:** this is the second real bug in the ingestion pipeline (after
the SIDER CID mismatch, #2) that live testing surfaced and a plausible-
looking, documented "verified fact" did not. Full-text search relevance
ranking from an external API should never be trusted as "first result is
correct" without an exact-match cross-check — regardless of how
confidently a prior verification step described the result.

---

## 6. Agent's run_prediction calls inference.py directly, not via HTTP —
   and a pytest-asyncio gotcha worth knowing before adding more DB tests

The Section 3 architecture diagram describes `run_prediction` as calling
"the FastAPI-served ML model", implying an HTTP round-trip. The agent
(`app/agent/predict.py`) instead calls `app/models/inference.py`'s
`predict_efficacy`/`predict_side_effects` functions directly in-process.
Reasoning: the agent and the FastAPI app both run in the same Python
process space and load the same joblib artifacts — routing through HTTP
would mean the agent making a network call to itself, adding a runtime
dependency (the API server must be up) for no isolation benefit. The
FastAPI layer (`app/api/`) is kept as a thin wrapper around the same
functions for any external caller that wants raw predictions without the
agent/report layer — it's not dead code, just not what the agent uses.

Separately: `app/db/session.py`'s `engine` is a module-level singleton,
which is correct for how the app actually runs (one process, one event
loop) but broke async DB tests under pytest-asyncio's default per-test
event loop scope (`InterfaceError: cannot perform operation: another
operation is in progress` on the second DB-touching test in a run — the
engine's connection pool stays bound to the first test's now-closed
loop). Fixed via `asyncio_default_fixture_loop_scope = "session"` /
`asyncio_default_test_loop_scope = "session"` in `pyproject.toml`, not by
changing the app's engine lifecycle. Worth knowing before adding more
async DB-touching tests — don't reach for per-test engine instances as a
"fix" for this; the session-scoped loop is the correct answer given the
app's actual singleton-engine design.

---

## 7. Local fine-tuned LLM pivot: base model choice, LoRA config, why only
   parse_query was fine-tuned, honest before/after results, RoPE/CLS
   framing, and the GGUF export that had to be abandoned mid-project

### Why this pivot happened

Two constraints pointed the same direction: no budget for continued
Anthropic API usage, and the project's stated primary goal (Section 2 of
the original continuation brief) is to demonstrate hands-on LLM
engineering — attention, fine-tuning, quantization — not just orchestrate
a hosted API. Local, open-weight models fine-tuned on the person's own
hardware (RTX 5070 Ti, 12GB VRAM, 32GB system RAM) solve both.

### Base models and why

- **`generate_report`**: `qwen2.5:14b-instruct` via Ollama, never
  fine-tuned. Inference-only, no training-memory constraint, no latency
  requirement (backend pipeline step, ~90-130s/report is fine) — the
  right call is to spend the available headroom on model quality, not to
  economize. **Not fine-tuned deliberately**: open-ended report prose has
  no gold-standard dataset to fine-tune against, and fabricating one
  would be worse than not fine-tuning at all (this exact reasoning was
  given up front, before any training happened — not a retroactive
  justification for skipping it).
- **`parse_query`**: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, QLoRA
  fine-tuned. This is the ONE node actually fine-tuned in this project.
  Kept at 7B rather than 14B deliberately: this was the one task under
  real time pressure on a multi-day timeline, and QLoRA
  OOM/instability risk on a bigger model is exactly the kind of thing
  that eats time a tight schedule doesn't have. (Considered and rejected
  swapping the allocation — 14B for parse_query, 7B for generate_report —
  mid-project: parse_query is a narrow, bounded-schema task where
  fine-tuning matters more than base-model scale, and the 7B fine-tune
  already reached 63% exact-field-match up from 30% base, so there was no
  evidence 14B would help enough to justify redoing a ~1hr fine-tune with
  higher OOM risk. generate_report is open-ended prose, exactly where
  scale shows up, and it's inference-only so the extra size costs
  nothing but generation time.)

### Why only parse_query was fine-tuned

`parse_query`'s job — extract compounds/patient/intent into a **bounded,
fixed JSON schema** — is tractable: the target shape is exactly
specifiable, so a synthetic (query → correct JSON) training set could be
generated programmatically with no manual labeling (see Step 0 and
`finetuning/generate_synthetic_data.py`). `generate_report` writes
open-ended prose with no equivalent gold-standard target. Fine-tuning a
model to imitate a fabricated "ideal report" dataset would have been
worse than leaving it on a strong instruct model with good prompting —
scope creep the pivot brief explicitly warned against.

### LoRA configuration (`finetuning/train_qlora.py`)

Rank 16, alpha 16 (alpha = rank is the conventional starting point),
dropout 0 (Unsloth's fast path is optimized for it), applied to
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. The
attention piece specifically: **q/k/v/o_proj ARE the query, key, value,
and output projection matrices of multi-head self-attention** — this is
the concrete, literal answer to "where does multi-head attention show up
in what you built." Learning rate 2e-4 (standard for LoRA — much higher
than full fine-tuning since only a small parameter subset updates), 3
epochs, effective batch size 8 (per-device 2 × gradient accumulation 4),
`adamw_8bit` optimizer (8-bit optimizer state, another memory saver
alongside 4-bit model weights).

### RoPE and attention: inherited, not implemented

Qwen2.5 uses RoPE (rotary position embeddings) and grouped-query
attention internally, from pretraining. **Fine-tuning does not let you
apply or configure RoPE — you get whatever the base model was pretrained
with.** The honest description of what this project demonstrates is:
choosing a model that uses RoPE and explaining what it does (relative
position encoding baked directly into the attention dot-product, via
rotating query/key vectors by an angle proportional to their sequence
position, giving better length generalization than learned absolute
position embeddings) — not "implementing" RoPE. Any code or claim that
implies otherwise would be wrong; this file exists partly so a viva
question about RoPE gets this answer, not an inflated one.

### CLS/MLM: BERT-family concepts, deliberately absent from the
   generative nodes

`parse_query` and `generate_report` are decoder-only generation tasks —
a CLS token and a masked-language-modeling objective are BERT-family
(encoder-only) concepts that don't fit either node, and no code in this
project adds one to them. The one place an encoder/CLS-pooled model
legitimately belongs is the Qdrant RAG retrieval step
(`app/ingestion/build_rag_index.py`), which already used FastEmbed's
`BAAI/bge-small-en` (BERT-derived, CLS-pooled, MLM-pretrained, then
further tuned for embedding quality) since Week 2 — verified, not newly
added for this pivot. That's the correct, motivated use of an
encoder-style model in this project: retrieval, not generation.

### Before/after comparison — the actual evidence, reported honestly

`finetuning/outputs/before_after_comparison.json` holds all 30 val
examples, both models' raw output, side by side. Summary:

| Metric | Base (7B, no fine-tune) | Fine-tuned |
|---|---|---|
| valid_json_rate | 97% | 80% |
| schema_valid_rate | 43% | **80%** |
| exact_field_match_rate | 30% | **63%** |

Training loss converged cleanly: 1.807 → 0.058 over 3 epochs (507 steps).
The `valid_json_rate` DROP (97%→80%) looks bad in isolation but is
explained, not hand-waved: **zero** fine-tuned examples were "valid JSON
but wrong schema" (vs. more than half the base model's valid-JSON outputs
being schema-wrong) — the fine-tune learned the target structure
essentially perfectly whenever it produced parseable JSON at all. The
gap is a narrow, specific formatting glitch: malformed list-bracket
syntax on the `comorbidities` field (`"comorbidities":s:["none"]` instead
of `"comorbidities":["none"]`) in ~20% of examples — not truncation
(outputs were well under the token limit), not a schema misunderstanding.
A single retry drops the theoretical compound failure rate toward ~1%,
but **live testing during Day 4 hit a query that failed on all 3 retry
attempts back to back** — report the retry as a mitigation, not a fix;
the failure mode is real and only partially bounded by retrying. The
graceful-degradation path in `app/agent/parse_query.py` (catch
`FinetunedParseError`, produce a clarification response) is what actually
guarantees the agent never crashes on this — not the retry count.

### The GGUF export had to be abandoned — direct in-process inference
   used instead

The original plan (matching `generate_report`'s pattern) was: merge the
LoRA into the base model, export to GGUF, serve via Ollama. The merge
step succeeded (produced a 15GB fp16 safetensors model, verified intact
on disk). **Converting that merged model to GGUF crashed the host
machine's IDE three separate times via kernel OOM** — the export's write
phase spiked memory faster than any monitoring-based safety net could
reliably catch (tried: no cap, a `systemd-run` cgroup `MemoryMax` cap,
and a 5-second self-kill watchdog; the kernel's own OOM killer won the
race against the watchdog on the third attempt, at 2.8GB available). Full
incident history in `finetuning/session_log/actions.log`.

**Resolution**: skip GGUF/Ollama entirely for `parse_query`'s fine-tuned
path. `app/agent/local_finetuned_model.py` loads the already-merged model
(the part that succeeded) directly via 4-bit quantization for in-process
inference — architecturally different from the crashing step because
it's **read-heavy** (memory-mapped safetensors, reclaimable page cache)
rather than **write-heavy** (dirty pages from writing a new 8-15GB file,
which need flushing before that memory is reclaimable — the actual
mechanism behind the crashes). Verified safe in two isolated test runs
(memory stable at ~23-25GB available throughout both) before being wired
into the real agent node and tested against all 5 required edge cases
end-to-end.

**VRAM tradeoff, disclosed**: this model needs ~5GB in 4-bit;
`generate_report`'s Ollama-served 14B model needs ~9GB. Together that
exceeds the 12GB card. Since the two nodes never run concurrently within
one agent pass, `local_finetuned_model.py` loads, generates, and
explicitly frees VRAM (`torch.cuda.empty_cache()`) on every single call
rather than keeping the model resident — trading ~35-40s of load latency
per `parse_query` call for the two models never fighting over VRAM. This
is the same "backend pipeline steps don't need to feel fast" tradeoff
already accepted for `generate_report`.

**Lesson for future sessions**: when a write-heavy operation is crashing
a shared/live environment, changing degree (smaller quant, tighter memory
cap, faster polling) is not the same as changing kind. The fix that
actually worked was recognizing that inference-loading and export-writing
are different memory access patterns, not tuning the write path harder.

---

*(Add further decisions below this line as they're made — model selection
rationale, feature schema finalization, etc.)*
