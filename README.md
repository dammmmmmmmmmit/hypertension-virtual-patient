# Virtual Patient Drug-Response Simulator

An LLM-augmented pipeline for hypertension drug-response screening: describe a
compound/patient question in plain language, and it resolves the compounds
against real pharmacology data (ChEMBL, PubChem, SIDER), runs a trained ML
prediction, and drafts a clinician-readable report — via a LangGraph agent
backed entirely by local, open-weight LLMs (no external API dependency).

**This is a screening/research tool, not clinical guidance.**

## Stack

FastAPI · Postgres · Redis · Qdrant · LangGraph · LightGBM · RDKit — backend.
Next.js, Tailwind, shadcn/ui, Motion — frontend. Ollama (`qwen2.5:14b-instruct`)
for report generation; a QLoRA-fine-tuned Qwen2.5-7B, served via direct
in-process 4-bit inference, for query parsing.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/)
- [Ollama](https://ollama.com/)
- Node.js 20+ / npm
- An NVIDIA GPU with ~12GB+ VRAM (for the fine-tuned model — see note below)

## Quick start

```bash
uv sync
cp .env.example .env

cd frontend && npm install && cd ..

ollama pull qwen2.5:14b-instruct

./run.sh
```

`run.sh` starts Postgres/Redis/Qdrant (creating the containers on first run),
Ollama, the FastAPI backend, and the Next.js frontend — safe to re-run any
time, it skips anything already up. Then open **http://localhost:3000**.

Stop the backend/frontend with `./stop.sh` (Docker containers and Ollama are
left running, since those are cheap to keep warm).

### Running it manually, one piece at a time

```bash
docker run -d -p 5433:5432 -e POSTGRES_PASSWORD=postgres --name vps-pg postgres:16
docker run -d -p 6379:6379 --name vps-redis redis:7
docker run -d -p 6333:6333 --name vps-qdrant qdrant/qdrant

ollama serve &

uv run uvicorn app.api.main:app --port 8000
# in another terminal:
cd frontend && npm run dev
```

The original Streamlit UI also still works, as a simpler single-process
alternative to the Next.js frontend:

```bash
uv run streamlit run app/ui/streamlit_app.py
```

## About the fine-tuned model weights

The QLoRA-fine-tuned `parse_query` model's merged weights (~15GB, fp16
safetensors) are **not included in this repo** — too large for git. To
reproduce them:

```bash
uv run python finetuning/train_qlora.py
```

This trains from the synthetic dataset already included in
`finetuning/data/` and produces the merged model at
`finetuning/parse_query_gguf/`, which `app/agent/local_finetuned_model.py`
loads at runtime. Without it, `parse_query` (and therefore the full pipeline)
won't run — everything else in the repo, including the trained LightGBM
models in `app/models/artifacts/`, is included and ready to use as-is.

## Tests

```bash
uv run pytest -q                                    # backend
cd frontend && npx tsc --noEmit && npx eslint .      # frontend
```

## Known limitations

- Efficacy predictions are anchored to class-level meta-analysis data with a
  small heuristic adjustment for compound potency — not real per-compound
  clinical outcomes.
- Side-effect model probabilities are consistently near-saturated regardless
  of drug (an artifact of a 14-row training set) — treat the ranking as
  meaningful, not the absolute value.
- Combination side-effect probabilities are an element-wise max across
  components, not a learned interaction.
- `parse_query`'s fine-tuned model has a known ~20% chance of a malformed-JSON
  output on any given call, mitigated by an internal retry, not eliminated.
- Patient covariate adjustments (renal function, etc.) are simplified
  disclosed multipliers, not validated PK/PD modeling.

## Project layout

```
app/
  schemas/       Pydantic models: patient, disease params, compounds
  core/          Drug registry, adjustment logic, static domain knowledge
  ingestion/     ChEMBL/PubChem/SIDER API clients + caching
  models/        ML training + inference
  agent/         LangGraph pipeline nodes
  api/           FastAPI routes
  ui/            Streamlit UI
frontend/        Next.js UI (Demo, Architecture, Domain, Data & methodology)
finetuning/      QLoRA fine-tuning pipeline for parse_query
tests/           Backend test suite
```
