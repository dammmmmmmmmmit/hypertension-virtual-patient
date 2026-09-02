# Project Context — Virtual Patient Drug-Response Simulator

Compiled as raw material for three downstream documents: (1) formal
documentation, (2) a technical article with diagrams/flowcharts, (3) a
personal command reference. Organized by topic so you can lift whole
sections into whichever document needs them. All numbers here are pulled
directly from the codebase, `DECISIONS.md`, `README.md`, live database
queries, and live pipeline test runs done this session — not from memory.

---

## 1. What this project is

**Virtual Patient Drug-Response Simulator** — an LLM-augmented pipeline,
scoped to **hypertension only**, that takes a natural-language
compound/patient description, resolves it against real pharmacology
databases (ChEMBL, PubChem, SIDER), runs a trained ML prediction, and
generates a clinician-readable report via a local LLM.

**Core goal (why it's built this way):** demonstrate equal engineering
rigor across every layer — data ingestion, ML, and LLM — not just a decent
model with an LLM bolted on for demo polish. This shows up throughout as a
refusal to fake precision anywhere: labels are honestly described as
semi-synthetic where they are, model confidence is disclosed as
uncalibrated where it is, and a hard clinical rule is kept as a rule
instead of dressed up as a model prediction.

**Explicit scope boundary, said everywhere in the app:** this is a
**screening/research tool, not point-of-care clinical guidance.** (Note:
the demo page's hero subtitle currently reads "An LLM-powered Clinical
Decision Support System for Hypertension Medication Analysis" per an
explicit content decision — flagged during the build as being in tension
with this positioning, kept anyway on request. Decide how you want to
frame this in formal documentation.)

**Why hypertension only:** modeling "a virtual patient" in general is out
of reach for a project this size. Five drug classes, blood
pressure/heart-rate/renal-function parameters, patient covariates as
adjustment factors rather than a full mechanistic physiology model.

---

## 2. Domain: five drug classes, fifteen drugs

Real class-level mean BP reduction at standard dose, from **Law MR,
Morris JK, Wald NJ. "Use of blood pressure lowering drugs in the
prevention of cardiovascular disease: meta-analysis of 147 randomised
trials..." BMJ 2009;338:b1665** (cross-checked against NIHR HTA
NBK62259). This is the **training label anchor**, not a per-compound
measurement.

| Class | Gene target | Systolic Δ (mmHg) | Diastolic Δ (mmHg) | Drugs (3 each) |
|---|---|---|---|---|
| ACE inhibitor | ACE | 8.5 | 4.7 | lisinopril, enalapril, ramipril |
| ARB | AGTR1 | 10.3 | 5.7 | losartan, valsartan, irbesartan |
| Beta-blocker | ADRB1 | 9.2 | 6.7 | metoprolol, atenolol, bisoprolol |
| Calcium-channel blocker | CACNA1C | 8.8 | 5.9 | amlodipine, nifedipine, diltiazem |
| Thiazide diuretic | SLC12A3 | 8.8 | 4.4 | hydrochlorothiazide, chlorthalidone, indapamide |

**Combination additivity** (same source): roughly additive but mildly
sub-additive — 1 drug → 4.7 mmHg (worked example, diastolic from 90
baseline), 2 drugs → 8.9 mmHg (94.7% of naive sum), 3 drugs → 12.6 mmHg
(89.4% of naive sum). Each added drug contributes ~95% of its
full-monotherapy effect relative to the previous step.

**The one hard-coded safety rule:** ACE inhibitor + ARB is flagged
discouraged — dual RAAS blockade, elevated hyperkalemia/renal-impairment
risk. This is read from `drug_registry.py`'s `DISCOURAGED_COMBINATIONS`,
**not a model output** — see §5 for why.

**Patient covariates tracked:** age, sex, weight, baseline systolic/
diastolic BP, heart rate, eGFR (renal function), serum potassium,
comorbidities (type 2 diabetes, chronic kidney disease, heart failure,
asthma/COPD), current medications. eGFR feeds a disclosed renal
adjustment multiplier applied *outside* the trained model, never baked
into training.

---

## 3. System architecture

### Stack
FastAPI · Postgres (asyncpg) · Redis (declared, not yet wired into the
agent) · Qdrant · LangGraph · LightGBM · RDKit · Ollama (local LLM
serving) · Unsloth/QLoRA (fine-tuning) · Next.js/Tailwind/shadcn (new
frontend) · Streamlit (original Week-2 UI, still present).

### The 7-stage agent pipeline (`app/agent/graph.py`)

Real LangGraph `StateGraph`, real node names, real order — verified
live via SSE streaming, not just read from source:

```
parse_query → resolve_entities → [conditional] → retrieve_data
  → structure_features → [conditional] → run_prediction
  → retrieve_comparators → generate_report
```

Conditional short-circuit branches: if entity resolution fails, or an
unsupported combination size is requested (only 1 or 2 drugs supported —
the training data doesn't cover 3+), the graph routes to a deterministic
clarification report instead of continuing. No stage is ever silently
skipped.

Real observed timing per stage:

| Stage | What it does | Timing |
|---|---|---|
| `parse_query` | Extract compounds/patient/intent from raw text — the ONE fine-tuned node | ~35–40s |
| `resolve_entities` | Match names against the drug registry (cached Postgres lookup) | sub-second |
| `retrieve_data` | Pull mechanism-of-action context from Qdrant RAG | sub-second |
| `structure_features` | Build the model's feature vector | sub-second |
| `run_prediction` | Run trained LightGBM models (in-process, not HTTP) | sub-second |
| `retrieve_comparators` | Look up standard-of-care comparator combos | sub-second |
| `generate_report` | Write the clinician-readable report (14B model via Ollama) | ~90–130s |

A full run end-to-end: **~2–3 minutes**, dominated entirely by the two
LLM-calling stages.

### Deliberate deviation from a naive architecture diagram
`run_prediction` calls `app/models/inference.py` functions **directly
in-process**, not over HTTP to the project's own FastAPI app — avoids a
pointless self-network-call. The FastAPI layer still exists and wraps the
same functions for any external caller that wants raw predictions.

### API surface
- `POST /predict/efficacy`, `/predict/side_effects` — thin wrappers over
  `inference.py`, for external callers.
- `POST /simulate/stream` — added this session for the new frontend.
  Wraps `graph.astream()` and re-emits each real per-node state update as
  a Server-Sent Event (`event: <node_name>`, `data: <json>`), plus a
  final `event: done`. This is what drives the frontend's live pipeline
  visualization — real backend events, not a client-side fake timer.

---

## 4. Data layer

### Sources
- **ChEMBL** — bioactivity/potency records (`pchembl_value`), mechanism
  text, SMILES structures.
- **PubChem** — CID resolution, cross-validated RDKit descriptors.
- **SIDER** (`sideeffects.embl.de`) — side-effect labels mined from FDA
  drug-label text. Joined by **drug name** against SIDER's own
  `drug_names.tsv`, not by independently re-deriving a PubChem-CID-based
  STITCH ID (that approach broke on lisinopril — PubChem's own name
  search resolves to a different stereoisomer CID than SIDER used).
- **RDKit** — molecular descriptors computed from ChEMBL's SMILES,
  cross-validated against ChEMBL's own precomputed properties (exact
  match).

### Real example record (losartan, from live Postgres query)
```
name                     losartan
drug_class               arb
gene_symbol              AGTR1
chembl_id                CHEMBL191
target_chembl_id         CHEMBL227
canonical_smiles         CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1
mean_potency (pX)        7.98
n_valid_potency_records  30
pubchem_cid              3961
```

### Real raw SIDER rows (losartan, STITCH ID CID100003961)
```
CID100003961  CID000003961  C0000737  LLT  C0000737  Abdominal pain
CID100003961  CID000003961  C0000737  PT   C0687713  Gastrointestinal pain
CID100003961  CID000003961  C0001824  LLT  C0001824  Agranulocytosis
```

### Two genuine, disclosed data gaps
- **Thiazide potency: 0/3 drugs.** Hydrochlorothiazide, chlorthalidone,
  indapamide all have `n_valid_potency_records = 0` in ChEMBL (thiazides
  act on a transporter, not the receptor-binding assay type ChEMBL is
  built around). Handled by setting potency z-score to 0 ("assume
  class-average"), not by fabricating a number.
- **Enalapril: 0 SIDER side effects.** 14/15 registry drugs have SIDER
  coverage; enalapril returns empty. Verified as a genuine SIDER gap, not
  a join-method artifact (checked directly, since the same name-based
  join correctly resolves the other 14).

### Two real bugs found by live testing, not code review
1. **Losartan resolved to the wrong compound.** ChEMBL's fuzzy full-text
   search (`/molecule/search.json?q=losartan`) returned `CHEMBL382821`
   ("LOSARTAN NITROOXY ESTER") — a distinct NO-donating derivative — as
   its top hit, not real losartan (`CHEMBL191`). Fix:
   `search_molecule_by_name()` now tries an exact `pref_name__iexact`
   match first, falling back to fuzzy search only when no exact match
   exists.
2. **Side-effect vocabulary dominated by near-universal terms.** Initial
   top-N-by-frequency vocabulary selection picked terms present in 14/14
   drugs regardless of mechanism (an artifact of how FDA drug labels get
   written, not real signal) — a classifier trained on an all-1 column
   learns nothing useful. Found because two pharmacologically unrelated
   drugs (atenolol, hydrochlorothiazide) returned byte-identical top-10
   side-effect predictions at ~0.9999999 probability. Fix: filter to
   terms present in at least 2 but no more than (n−2) drugs before
   ranking by frequency.

---

## 5. ML models

### Two trained tasks, not three
The original plan specified three: efficacy, side-effect probability,
and a standalone "combination interaction" classifier. Rejected: with
only 5 drug classes, there are exactly C(5,2) = 10 possible class-pairs,
of which exactly 1 is discouraged — a classifier on 10 rows with 1
positive would be memorization, not ML.

**Resolution:** combination-interaction signal folded into task 1
(efficacy) — its training set includes both single-drug and 2-drug
combination rows, so the model has to learn the difference directly.
`DISCOURAGED_COMBINATIONS` stays a **hard-coded rule**, not a model
output — presenting a fixed clinical fact with a model confidence score
would *understate* how well-established the risk is.

**Net result:** 2 trained ML tasks (LightGBM efficacy regression,
LightGBM side-effect multi-label classification) + 1 rule-based safety
check.

### Efficacy label construction (semi-synthetic — the single biggest
honesty tradeoff in the project)
No public per-compound RCT dataset exists at consistent dosing across all
15 drugs, so the label is built, not measured:

```
baseline_delta(class)   = class-level trial mean (Law/Morris/Wald, BMJ 2009)
potency_z(drug)         = (drug's pX − class mean pX) / class std pX
                           (thiazides: z = 0, "assume class-average")
adjusted_delta(drug)    = baseline_delta(class) * (1 + ALPHA * clip(potency_z, −2, 2))
                           ALPHA = 0.08 — hand-set, NOT fit from data
```
Combination rows apply the empirical ~95%-per-added-drug sub-additivity
discount, then the disclosed renal adjustment factor on top (never baked
into training).

### Real training data
- **Efficacy:** 60 rows (15 single-drug + 45 combinations).
- **Side-effect:** 14 rows (one per drug with SIDER coverage) × 29-label
  filtered vocabulary.

### Real evaluation numbers (leave-one-out CV — small n, no held-out
split is meaningful)
| Model | n | Metric | Value | Honest caveat |
|---|---|---|---|---|
| Efficacy | 60 | MAE | 0.171 mmHg (SBP) / 0.295 mmHg (DBP) | Labels are semi-synthetic — low error reflects recovering a smooth constructed function, not real-world predictive accuracy. |
| Side-effect | 14 | Mean per-label accuracy | 0.729 | Inflated by near-constant labels (a side effect in 13/14 drugs is "accurate" even predicting the majority class). Treat as a smoothed lookup over structural similarity, not a validated QSAR model. |

### Known, disclosed limitations (carry these into any documentation —
don't soften them)
- Efficacy predictions are class-anchored + small heuristic adjustment,
  not real per-compound clinical outcomes.
- Side-effect probabilities are consistently near-saturated (~0.99+
  regardless of drug) — an artifact of n=14 training rows. **Ranking is
  meaningful; the absolute value is not.** (This was actively causing a
  UX problem — see §7's honesty fix.)
- Combination side-effect probability = element-wise max across
  components, not a learned interaction (no real combo-level SIDER data
  exists).
- Patient covariate adjustments (renal function, etc.) are simplified
  disclosed multipliers, not validated PK/PD modeling.

---

## 6. LLM layer: the local pivot

### Why local models
Two constraints: no budget for continued Anthropic API usage, and the
project's actual goal (demonstrate hands-on LLM engineering — attention,
fine-tuning, quantization — not just orchestrate a hosted API). Both LLM
nodes now run entirely locally; there is no Anthropic API dependency left
anywhere in the app (the `ANTHROPIC_API_KEY` line still present in
`.env.example` is vestigial from before the pivot).

### Two nodes, two very different treatments

| | `generate_report` | `parse_query` |
|---|---|---|
| Model | `qwen2.5:14b-instruct` via Ollama | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, QLoRA fine-tuned |
| Fine-tuned? | **No** — deliberately | **Yes — the ONE fine-tuned node** |
| Why | Open-ended report prose has no gold-standard dataset to fine-tune against; fabricating one would be worse than not fine-tuning. Inference-only, no training-memory constraint, no tight latency requirement → spend headroom on model quality (14B) instead. | Bounded, exactly-specifiable JSON schema → a synthetic (query → correct JSON) training set could be generated programmatically. Kept at 7B (not 14B) deliberately: real time pressure, and the 7B fine-tune already reached 63% exact-field-match from a 30% base — no evidence 14B would help enough to justify redoing the ~1hr fine-tune with higher OOM risk. |
| Serving | Ollama | **Direct in-process 4-bit inference** — NOT Ollama/GGUF (see below) |

### LoRA configuration (`finetuning/train_qlora.py`)
Rank 16, alpha 16 (alpha = rank, conventional starting point), dropout 0,
applied to `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj`. **q/k/v/o_proj are the literal query/key/value/output
projection matrices of multi-head self-attention** — the concrete answer
to "where does attention show up in what you built." LR 2e-4, 3 epochs,
effective batch size 8 (2 per-device × 4 gradient accumulation),
`adamw_8bit` optimizer. Training loss: 1.807 → 0.058 over 3 epochs (507
steps).

### RoPE and attention: inherited, not implemented (important for a viva)
Qwen2.5 uses RoPE (rotary position embeddings) and grouped-query
attention internally, from pretraining. **Fine-tuning does not let you
apply or configure RoPE** — you get whatever the base model was
pretrained with. What this project actually demonstrates: choosing a
model that uses RoPE and explaining what it does (relative position
encoding baked into the attention dot-product by rotating query/key
vectors by an angle proportional to sequence position), and training LoRA
adapters that target the attention projection matrices directly. Not
"implementing" RoPE — any claim that implies otherwise would be wrong.

### CLS/MLM: deliberately absent from the generative nodes
`parse_query` and `generate_report` are decoder-only generation tasks — a
CLS token / masked-language-modeling objective are BERT-family
(encoder-only) concepts that don't fit either. The one place an
encoder/CLS-pooled model legitimately belongs: the Qdrant RAG retrieval
step, which uses FastEmbed's `BAAI/bge-small-en` (BERT-derived,
CLS-pooled, MLM-pretrained) — used since Week 2, unrelated to the pivot.

### Before/after fine-tuning — honest numbers
(`finetuning/outputs/before_after_comparison.json`, 30 val examples)

| Metric | Base (7B, no fine-tune) | Fine-tuned |
|---|---|---|
| valid_json_rate | 97% | 80% |
| schema_valid_rate | 43% | **80%** |
| exact_field_match_rate | 30% | **63%** |

The `valid_json_rate` *drop* looks bad in isolation but isn't: **zero**
fine-tuned examples were "valid JSON but wrong schema" (vs. more than
half of base's valid-JSON outputs being schema-wrong) — the fine-tune
learned the target structure almost perfectly whenever it produced
parseable JSON at all. The gap is one narrow bug: malformed
list-bracket syntax on the `comorbidities` field in ~20% of examples. A
retry brings the theoretical failure rate toward ~1%, but live testing
hit a query that failed all 3 retries back to back — report the retry as
a mitigation, not a fix. The agent's graceful-degradation path (a
clarification response, never a crash) is what actually guarantees
robustness.

### The GGUF export crash story
Original plan (matching `generate_report`'s pattern): merge LoRA into
base, export to GGUF, serve via Ollama. The merge succeeded (15GB fp16
safetensors, verified intact). **Converting to GGUF crashed the host
machine's IDE three separate times via kernel OOM** — tried no cap, a
`systemd-run` cgroup `MemoryMax` cap, and a 5-second self-kill watchdog;
the kernel's own OOM killer won the race on the third attempt.

**Resolution:** skip GGUF/Ollama entirely for `parse_query`.
`app/agent/local_finetuned_model.py` loads the already-merged model
directly via 4-bit quantization for in-process inference — architecturally
different because it's **read-heavy** (memory-mapped safetensors,
reclaimable page cache) not **write-heavy** (dirty pages from writing a
new 8–15GB file, the actual OOM mechanism). Verified stable (23–25GB
available throughout) in two isolated test runs before wiring into the
real agent node.

**Lesson worth stating explicitly in documentation:** when a write-heavy
operation is crashing a shared environment, changing degree (smaller
quant, tighter cap, faster polling) is not the same as changing kind. The
fix that worked was recognizing these are different memory access
patterns, not tuning the write path harder.

**Disclosed VRAM tradeoff:** fine-tuned model needs ~5GB in 4-bit;
`generate_report`'s Ollama 14B model needs ~9GB — together over the 12GB
card (RTX 5070 Ti). Since the two nodes never run concurrently,
`local_finetuned_model.py` loads, generates, and explicitly frees VRAM
(`torch.cuda.empty_cache()`) on every call rather than staying resident —
trading load latency (part of the ~35–40s) for the two models never
fighting over VRAM.

### A real architecture bug found and fixed this session
`parse_query.py` used to branch: full extraction via the fine-tuned model
when no patient was pre-supplied, but a *different*, non-fine-tuned
base-model path for compounds+intent-only extraction when a patient
*was* pre-supplied (e.g. the Streamlit structured form, and now the new
frontend's patient form — i.e. the one flow every real UI actually uses).
That meant the fine-tuned model was silently never exercised in practice,
contradicting documentation and making the pipeline-visualization timing
wrong. Found via live end-to-end SSE testing of the new frontend, not
code review. **Fixed:** always call the fine-tuned model; when a real
patient profile is already supplied, its (unverified) patient-guess
portion is simply discarded in favor of the caller-supplied one.

---

## 7. A live-testing-driven honesty fix worth including in the article

While building the frontend, a real run showed the report literally
stating "Cough: Probability = 1.00" for **eight different** side effects
in a row — technically defensible (raw values were ~0.999x, rounded to
2dp) but read as absurd to any viewer, and traced to `_format_context()`
in `generate_report.py` handing the LLM pre-rounded 2dp percentages
computed from an inherently near-saturated, uncalibrated model (n=14
training rows — see §5).

**Two-part fix:**
1. **Backend prompt fix** (root cause, benefits every consumer including
   Streamlit): stopped feeding the LLM literal percentages; instructed it
   to describe side effects by rank only ("most prominently predicted...
   also predicted, though lower-ranked...") with an explicit calibration
   caveat.
2. **Frontend structured panel**: a `SideEffectPanel` component reads the
   real probability dict directly (not LLM prose) and renders rank-tier
   badges (Top/Mid/Lower-ranked, derived from ordinal position, not raw
   magnitude) and an evidence badge (single-compound vs.
   combination/max-combined) — deliberately labeled "relative rank," not
   "relative risk," since true relative risk is a specific epidemiological
   ratio this project has no baseline data to compute.

Good case study for the article: a plausible-looking, technically-correct
number that was nonetheless misleading, caught by actually looking at
live output rather than trusting the pipeline compiled and ran.

---

## 8. Frontend (Next.js — new this session, alongside the original Streamlit UI)

### Stack
Next.js 16.3.0 (App Router, Turbopack default), React 19.2.8, TypeScript,
Tailwind CSS v4 (CSS-first `@theme` config, no `tailwind.config.js`),
shadcn/ui (style `base-nova`, Base UI primitives), Motion (`motion`
package, successor to Framer Motion), react-markdown.

### Color system — "Electropop," one job per color
- **Acid** `#ccff00` — primary interactive accent. ONE most-important
  element per view only (hero CTA, active pipeline stage, focus rings).
- **Indigo** `#5200ff` — structural/quiet chrome (nav, dividers).
- **Magenta** `#f900ff` — discouraged-combination warning state ONLY.
  Mapped to shadcn's `destructive` token.
- **Caveat orange** `#ff6b00` — "attention, not danger" (data-sparsity
  flags, caveats). Distinct from magenta on purpose.

### Four pages
1. **Demo** (`/`) — free-text query + structured patient-vitals form
   (mirrors the Streamlit sidebar's fields/defaults exactly) → live
   pipeline visualization → generated report.
2. **Architecture** (`/architecture`) — the 7-stage pipeline diagram, the
   local-LLM pivot story, LoRA config, before/after table, RoPE/CLS
   honesty framing, "two tasks not three."
3. **Domain** (`/domain`) — five drug classes with real BMJ figures, the
   discouraged-combination rule, patient covariates.
4. **Data & methodology** (`/data`) — real ChEMBL/SIDER records pulled
   from the live cache, the two genuine data gaps, real training rows,
   the efficacy label formula, honest evaluation numbers.

### The pipeline visualization (the "signature" piece, built last)
Driven entirely by real SSE events from `/simulate/stream` via a
`useSimulation` hook — elapsed times per stage are real `Date.now()`
deltas recorded as events arrive, not fabricated progress. The two slow
LLM stages show rotating "likely activity" text grounded in real
implementation details (4-bit model load, the 5 mandated report
sections, the retry-on-malformed-JSON path), explicitly labeled on-page
as illustrative, not a live trace (the backend only emits one event per
whole-node completion). Respects `prefers-reduced-motion` throughout.

### Backend integration notes
- `POST /simulate/stream` (new this session) streams real LangGraph node
  completions as SSE, consumed via `fetch()` + manual `ReadableStream`
  parsing (not native `EventSource`, which only supports GET — this
  endpoint needs a JSON POST body).
- CORS added to `app/api/main.py` for `http://localhost:3000`.
- The Streamlit UI (`app/ui/streamlit_app.py`, Week 2) is the original
  reference implementation and remains functional — same backend, same
  `run_agent()` entry point.

---

## 9. Testing & verification discipline

- **77 pytest tests**, all passing. `asyncio_default_fixture_loop_scope
  = "session"` / `asyncio_default_test_loop_scope = "session"` in
  `pyproject.toml` — required because `app/db/session.py`'s `engine` is a
  correct module-level singleton for how the app actually runs, but
  breaks under pytest-asyncio's default per-test event loop.
- Every major pipeline claim in this document was **live-verified**, not
  just read from source: real curl tests against `/simulate/stream` for a
  single drug, a standard combo, and the discouraged ACE+ARB combo, with
  full SSE event sequences and report content inspected.
- Frontend: `tsc --noEmit` and `eslint` clean throughout; every page
  fetched and confirmed rendering (HTTP 200 + expected content) after
  each change.
- This project's own running theme, worth stating explicitly in
  documentation: **several real bugs were found only by live-testing
  actual output, not by code review** — the losartan mismatch, the
  side-effect vocabulary saturation, the parse_query branch bug, and the
  side-effect-probability display issue were all caught this way.

---

## 10. Suggested diagrams/flowcharts for the article

1. **End-to-end pipeline flow** — the 7 LangGraph nodes in sequence, with
   the two conditional short-circuit branches to the clarification
   report, and real timing annotated per node.
2. **Data ingestion flow** — ChEMBL / PubChem / SIDER → resolution &
   caching (Postgres `resolved_compounds`) → feature engineering →
   training datasets. Annotate the two real data gaps (thiazide potency,
   enalapril SIDER) where they enter.
3. **Model architecture** — two LightGBM heads (efficacy regression,
   side-effect multi-label classification) fed from shared features, with
   the `DISCOURAGED_COMBINATIONS` rule-based check drawn as a separate,
   parallel, non-ML path.
4. **LLM serving architecture, before vs. after the pivot** — Anthropic
   API (single box) → two local paths: Ollama-served 14B
   (`generate_report`) and direct in-process 4-bit QLoRA-fine-tuned 7B
   (`parse_query`), with the abandoned GGUF/Ollama path shown crossed out
   and annotated with the crash reason.
5. **LoRA/attention diagram** — a standard transformer block with
   q/k/v/o_proj and gate/up/down_proj highlighted as the adapted
   matrices, to make the "where does fine-tuning touch attention"
   question concrete.
6. **Frontend SSE data flow** — browser `fetch()` POST → FastAPI
   `StreamingResponse` wrapping `graph.astream()` → per-node SSE events →
   `useSimulation` hook accumulating state → pipeline visualization +
   report display.
7. **Side-effect honesty fix, before/after** — a small before/after pair
   showing the old literal "Probability = 1.00" text next to the new
   rank-based structured panel, as a concrete case study of the project's
   "verify against live output" discipline.

---

## 11. Commands reference

### One-time setup
```bash
cd /home/as/Desktop/college/LLM/virtual_drug_sim/vps
uv sync
cp .env.example .env   # DATABASE_URL/REDIS_URL/QDRANT_URL — ANTHROPIC_API_KEY is vestigial, unused post-pivot

cd frontend
npm install
# .env.local already present: NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### Start infrastructure (Postgres on 5433, not 5432 — avoids conflict
with an unrelated project's container on this machine)
```bash
docker run -d -p 5433:5432 -e POSTGRES_PASSWORD=postgres --name vps-pg postgres:16
docker run -d -p 6379:6379 --name vps-redis redis:7
docker run -d -p 6333:6333 --name vps-qdrant qdrant/qdrant

# subsequent sessions:
docker start vps-pg vps-redis vps-qdrant
docker ps --format "{{.Names}}: {{.Status}}" | grep vps
```

### Ollama (serves generate_report's 14B model)
```bash
ollama serve &
ollama pull qwen2.5:14b-instruct
pgrep -af "ollama serve"          # verify it's running
```

### Run the backend
```bash
cd /home/as/Desktop/college/LLM/virtual_drug_sim/vps
uv run uvicorn app.api.main:app --port 8000
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/docs   # verify: 200
```

### Run the frontend
```bash
cd /home/as/Desktop/college/LLM/virtual_drug_sim/vps/frontend
npm run dev
# open http://localhost:3000
```

### Run the original Streamlit UI (alternative to the Next.js frontend)
```bash
cd /home/as/Desktop/college/LLM/virtual_drug_sim/vps
uv run streamlit run app/ui/streamlit_app.py
```

### Tests
```bash
uv run pytest -q                                    # backend, 77 tests
cd frontend && npx tsc --noEmit && npx eslint .      # frontend
```

### Rebuild data / models (only if starting from scratch or data changed)
```bash
uv run python -m app.ingestion.build_registry      # ChEMBL/PubChem resolution
uv run python -m app.ingestion.populate_cache       # populate resolved_compounds
uv run python -m app.ingestion.build_rag_index      # Qdrant mechanism-text index
uv run python -m app.models.train                   # train + save LightGBM artifacts
```

### Fine-tuning (parse_query's QLoRA adapter — already trained; only
needed to reproduce)
```bash
uv run python finetuning/generate_synthetic_data.py
uv run python finetuning/train_qlora.py
uv run python finetuning/test_direct_inference.py    # sanity check before wiring into the agent
```

### Useful direct checks
```bash
# Full pipeline via curl (mirrors what the frontend does)
curl -s -N -X POST http://localhost:8000/simulate/stream \
  -H "Content-Type: application/json" \
  -d '{"raw_query":"How would 50mg losartan affect this patient'"'"'s blood pressure?",
       "patient":{"age":58,"sex":"male","weight_kg":82,
       "baseline":{"systolic_bp":152,"diastolic_bp":96,"heart_rate":78,"serum_potassium":4.2,"egfr":90},
       "comorbidities":["none"],"current_medications":[]}}'

# Inspect a real cached compound record
docker exec vps-pg psql -U postgres -d vps -c \
  "SELECT name, chembl_id, mean_potency, n_valid_potency_records FROM resolved_compounds WHERE name='losartan';"
```

---

## 12. Key files to cite directly in documentation

- `DECISIONS.md` — the full engineering decisions log (8 entries), single
  source of truth for every non-obvious call made in this project.
  Everything in §5–7 above is condensed from here.
- `README.md` — status tracker, known limitations, honesty guardrails.
- `app/agent/graph.py` — the real pipeline definition.
- `app/agent/generate_report.py` — the report-writing prompt (recently
  patched for the side-effect honesty fix).
- `app/models/artifacts/training_metrics.json` — real LOO-CV numbers.
- `finetuning/outputs/before_after_comparison.json` — the fine-tuning
  before/after raw data.
- `finetuning/session_log/actions.log` — chronological log of the GGUF
  crash incident, if you want that level of detail for the article.
