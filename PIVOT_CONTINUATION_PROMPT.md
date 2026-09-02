# Continuation Brief: Local Fine-Tuned LLM Pivot

## Role

You are implementing an architectural pivot on an existing, working, extensively-tested codebase — the hypertension virtual patient drug-response simulator. This is NOT a restart. Two weeks of real engineering work already happened: a validated ML core, a working LangGraph agent, a tested FastAPI layer, 77 passing tests, and a 6-entry `DECISIONS.md` documenting hard-won fixes to real bugs. **Read `DECISIONS.md` in full, then read the actual current repo files, before writing or changing anything.** That file — not this brief, and not your own general knowledge — is the authoritative record of what's already been decided and why. Where this brief and `DECISIONS.md` ever seem to conflict, `DECISIONS.md` describing the real, current repo wins; flag the conflict rather than silently picking one.

This brief covers ONE thing: replacing the Anthropic-API-based LLM layer (`parse_query`, `generate_report`) with local, open-weight models, and fine-tuning one of them. Everything else about the project — the ML core, the efficacy-label design, the drug registry, the RAG index's content, the discouraged-combination rule, the agent graph structure — is unchanged and must not be "helpfully" redesigned in the course of this pivot.

## Why this pivot is happening

Two constraints turned out to have the same answer:
1. **No budget** for Anthropic API usage.
2. **The project's stated primary goal** (see the original continuation brief, Section 2) is to demonstrate LLM engineering skill, and the person specifically wants hands-on demonstration of transformer/LLM foundations — multi-head self-attention, positional encoding, fine-tuning, parameter tuning — not just orchestration around a hosted API.

Local, open-weight models fine-tuned on the person's own hardware (RTX 5070 Ti, 12GB VRAM) solve both at once: free inference, and weights you can actually fine-tune and explain.

## Critical technical framing — internalize this before touching any model/architecture choice

**CLS tokens and MLM (masked language modeling) do not belong in this pipeline's generative nodes.** They're BERT-family (encoder-only) concepts; `parse_query` and `generate_report` are decoder-only generation tasks. Do not add a CLS token or an MLM objective anywhere in the fine-tuning of either node — it would be architecturally incoherent, not just unnecessary. The ONE place an encoder-style model legitimately belongs in this project is the **Qdrant RAG retrieval embeddings** (see below) — that's a real, correctly-motivated use of a BERT-derived, CLS-pooled model, and it's the only place one should appear.

**RoPE is inherited, not implemented.** Whatever open decoder model you choose (Qwen2.5/3, Llama 3.1, etc.) already uses RoPE internally from pretraining. Fine-tuning does not let you "apply" or choose RoPE vs. NoPE — you get whatever the base model was pretrained with. The honest way to demonstrate understanding of this is a clear written explanation (in the final docs, see Day 5) of what RoPE does and why it matters, not a code change. Do not write or suggest code that implies RoPE is being newly implemented or configured here.

**Only `parse_query` gets fine-tuned. `generate_report` does not.** `parse_query` (NL → bounded structured JSON) is tractable: the schema is bounded, and a synthetic training set can be generated programmatically with no manual labeling. `generate_report` (open-ended prose synthesis) has no gold-standard report dataset to fine-tune against — fabricating one would be worse than not fine-tuning at all. `generate_report` stays on a strong local instruct model with good prompting, unmodified in weights. If you find yourself building a report-fine-tuning dataset, stop — that's scope creep the person explicitly didn't ask for and can't cheaply verify the quality of.

**A fine-tune without a documented before/after comparison is not evidence it worked.** Every fine-tuning result must be reported alongside the base (non-fine-tuned) model's output on the same validation inputs. This is not optional polish — it's the actual evidence for the project's core claim ("I fine-tuned a model and it helped"). If the improvement is small, report the small improvement honestly; that is more credible than an inflated claim, consistent with how `DECISIONS.md` #4 handled the constant-label bug and #1 handled the semi-synthetic efficacy label caveat.

**The two LLM nodes have different hardware headroom — don't treat them identically.** Confirmed hardware: RTX 5070 Ti (12GB VRAM), 32GB system RAM, capable CPU. `generate_report` is inference-only with no training-memory constraint and no latency requirement (backend pipeline step, not live chat) — default to a 14B model there, a 30B-class MoE with CPU/RAM offload is a reasonable stretch to try. `parse_query`'s fine-tuning, by contrast, is the one task under real time pressure on this 3-5 day timeline — default stays 7-8B for training reliability; `train_qlora.py` has a clearly-marked stretch-option block (13-14B + `paged_adamw_8bit` + batch size 1) for later if the 7B fine-tune's before/after comparison looks weak and there's a spare day to spend on it. Don't default to the stretch option on the fine-tuned node just because the hardware technically allows it — that trades schedule risk for a benefit that's unproven until the 7B baseline result is actually in hand.

## What already exists from a prior planning session (not yet integrated into the repo)

A previous session (not this one) produced a zip the person has already unzipped into the repo root, adding a `finetuning/` directory alongside the existing `app/`. Contents, with an honest status on each:

- **`finetuning/generate_synthetic_data.py`** — TESTED AND WORKING. Ran end-to-end, produced 1500 examples (1350 train / 150 val), verified: 0 formatting artifacts (a real bug — a placeholder-cleanup approach that silently left `?` in ~14% of examples — was caught by testing and fixed by building optional clauses compositionally instead), 100% valid JSON, balanced across 4 question-intent classes and single/standard-combo/discouraged-combo/thiazide-combo cases. **However, it imports from `app/schemas/reference.py` and `app/core/reference_registry.py` — RECONSTRUCTED placeholder schemas, not your real ones.** This is Step 0 below, not optional.
- **`finetuning/train_qlora.py`** — UNTESTED. No GPU was available in the sandbox that wrote it. Written carefully against Unsloth's standard QLoRA pattern (4-bit load, LoRA on q/k/v/o_proj + MLP projections, `SFTTrainer`, GGUF export), but treat it exactly like `chembl_client.py` was treated on Day 2 of the original build: a solid first draft that needs debugging against reality, not verified-working code.
- **`finetuning/Modelfile.parse_query`** — UNTESTED, and explicitly flagged inside the file: the `FROM` line's filename is a guess at what Unsloth's GGUF export produces; verify against the actual output directory contents before trusting it. The chat TEMPLATE must exactly match whatever `tokenizer.apply_chat_template()` produced during training, or serving and training will silently disagree.
- **`finetuning/PLAN.md`** — the day-by-day plan this brief is built on top of; read it, it has detail this brief summarizes.
- **`app/schemas/reference.py`, `app/core/reference_registry.py`** — RECONSTRUCTED, NOT REAL. These exist only so the data generator had something to import and could be tested in a sandbox without access to the real repo. Your real `app/agent/schemas.py` and `app/core/drug_registry.py` have evolved past these over two weeks of work this planning session didn't see.

## Step 0 — do this before anything else

1. Read the real `app/agent/schemas.py` (or wherever the current `SimulationRequest`/`CompoundQuery`/patient schema now lives — check `app/agent/state.py` too) and the real `app/core/drug_registry.py`.
2. Reconcile `generate_synthetic_data.py`'s imports and target-JSON field names against these real schemas — field names WILL have drifted from the reconstructed reference (e.g. the real schema may have renamed fields, added the `drug_class`/`chembl_id` fields that were stripped from the reference version, etc.).
3. Regenerate the dataset against the real schema. Re-verify the same three things the original testing pass checked: 0 formatting artifacts, 100% valid JSON, balanced distribution. Do not skip re-verification just because the generator logic itself was already tested — the logic was tested against placeholder schemas, not real ones.
4. Only then proceed to Day 1 below.

## Day-by-day plan

**Day 1 — lowest-risk win first: swap `generate_report` to local Ollama, no fine-tuning needed.**
Hardware confirmed: RTX 5070 Ti (12GB VRAM), 32GB system RAM, capable CPU. `generate_report` is inference-only and never fine-tuned (see above) — it has no training-memory constraint, and as a backend pipeline step (not live chat) it doesn't need to feel fast, so this is the node to be ambitious on model size. Install Ollama, pull `qwen2.5:14b-instruct` (default — fits fully in VRAM at 4-bit, meaningfully better prose than a 7B model, no offload complexity). If report quality still feels thin, a 30B-class MoE model split across VRAM+RAM via Ollama's automatic layer offload is worth trying — slower generation, but that's an acceptable tradeoff here specifically; benchmark actual tokens/sec on the real machine before committing to it, don't assume it'll be fast enough. Swap the LLM backend in the real `generate_report` node from `ChatAnthropic` to `ChatOllama` (`langchain-ollama`) — same LangChain interface, should be close to a drop-in swap if structured output / `.invoke()` is used consistently already. Re-run the existing report-generation tests. Compare output quality against the Claude-generated reports honestly and document the difference.

**Day 2 — after Step 0's regeneration, inspect the dataset by hand.**
Read 20-30 examples for real: are typos realistic, is `question_intent` correctly labeled for the phrasing, does the JSON match the real schema exactly? Decide final dataset size (1500 is a reasonable start for this task's bounded scope).

**Day 3 — QLoRA fine-tuning.**
Install the fine-tuning stack, run `train_qlora.py`, watch VRAM the first run (drop batch size / raise gradient accumulation if near the 12GB ceiling). Check the script's val-set JSON-validity output BEFORE the loss curve — a pretty loss curve with broken JSON output is the same trap as `DECISIONS.md` #4's constant-label bug. **Run the same val inputs through the base (non-fine-tuned) model too and save both outputs side by side** — this comparison is mandatory, see "Critical technical framing" above. Export to GGUF.

**Day 4 — integration + the RAG embedding model.**
Build the Ollama model from the Modelfile (fix the `FROM` path first), swap `parse_query`'s backend to it the same way `generate_report` was swapped on Day 1. Re-run all 5 existing edge-case tests (single drug, standard combo, discouraged ACE+ARB combo, thiazide missing-potency combo, misspelled name) end-to-end through the full pipeline — these are regression tests against `DECISIONS.md`'s existing behavior, not new tests to design from scratch. Separately: if the current Qdrant retrieval isn't already using a real sentence-embedding model, add one (e.g. `BAAI/bge-small-en-v1.5` via `sentence-transformers`) and re-embed the MoA text chunks — this is the project's one legitimate encoder/CLS touchpoint, keep it scoped to retrieval only, don't let it expand into a bigger redesign.

**Day 5 — documentation + buffer.**
Add a new `DECISIONS.md` entry (follow the existing entries' format and rigor) covering: which base model and why, RoPE/attention explained honestly as inherited-not-implemented, why only `parse_query` was fine-tuned, the LoRA config choices (rank/alpha/target modules — q/k/v/o_proj ARE the multi-head attention projections, this is the concrete answer to "where does multi-head attention show up in what you built"), the before/after comparison results reported honestly even if the delta is modest, and the embedding-model note for retrieval. Use remaining time as buffer for debugging — GGUF export filenames, chat-template mismatches, and VRAM tuning are the most likely snags, in that order.

## How to start

Confirm you've read `DECISIONS.md` and the real current schema/registry files. Do Step 0. Then proceed day by day. At the end of each day, give a short status summary in the same style `DECISIONS.md` already uses — what works, what's unverified, what was skipped or descoped and why. Do not mark something done if it's untested, and do not silently drop the before/after comparison requirement or the CLS/RoPE framing above under time pressure — those are the parts of this pivot most likely to be quietly skipped when things get tight, and they're also the parts that matter most for the project's actual stated goal.
