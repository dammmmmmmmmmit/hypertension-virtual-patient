# Local Fine-Tuning Pivot: Day-by-Day Plan (3-5 days)

Scope: replace the Anthropic API in `parse_query` and `generate_report`
with local, open-weight models (free, runs on your RTX 5070 Ti), and
fine-tune the `parse_query` model via QLoRA to genuinely demonstrate
applied LLM engineering - not just call an API and call it done.

**What's tested vs. not, honestly:**
- `generate_synthetic_data.py` - tested in full, 1500 examples generated
  and verified (valid JSON, 0 formatting artifacts after a bug I caught
  by testing and fixed, balanced across question intents and combo types)
- `train_qlora.py` - CANNOT be tested here (no GPU, no HF network access
  in this sandbox). Written carefully against Unsloth's standard pattern,
  but treat it as a first draft you'll debug on your own machine, same
  as chembl_client.py was on Day 2 of the original build.
- Ollama Modelfile - filename/template details need verification against
  what your training run actually produces.
- Integration patterns below are PATTERNS, not edits to your real files -
  I don't have your current app/agent/parse_query.py or
  generate_report.py (they've evolved past my original scaffold over
  two weeks of real work). Adapt field/import names to match your
  actual code.

---

## Day 1: Environment + immediate win (swap generate_report first)

1. Install Ollama (https://ollama.com), then:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
2. Swap `generate_report`'s LLM backend from Anthropic to Ollama. This
   needs NO fine-tuning - a decent instruct model handles open-ended
   report generation fine with good prompting. Pattern (LangChain
   exposes the same interface across providers, so this should be close
   to a drop-in swap if you're using `with_structured_output` / `.invoke`
   the same way):

   ```python
   # before:
   # from langchain_anthropic import ChatAnthropic
   # llm = ChatAnthropic(model="claude-sonnet-5", temperature=0.7)

   # after:
   from langchain_ollama import ChatOllama
   llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.7)
   ```
   `pip install langchain-ollama --break-system-packages`

3. Re-run your existing report-generation tests (the 5 edge cases) against
   this local model. Compare output quality against your Claude-generated
   reports honestly - note the difference in your writeup rather than
   hiding it. A 7B local model will likely be noticeably weaker at prose
   quality than Sonnet; that's an expected, disclosable tradeoff for
   "runs free and local," not a bug to hide.

This alone gets you off the API for one whole node on day 1, with zero
fine-tuning risk yet.

## Day 2: Generate and inspect the synthetic training data

1. Copy `generate_synthetic_data.py` into your repo (e.g.
   `finetuning/generate_synthetic_data.py`), fix the import at the top to
   pull from your REAL `app/agent/schemas.py` and drug registry instead
   of the reconstructed reference versions.
2. Run it, read through 20-30 examples by hand before trusting it -
   exactly the same discipline as everywhere else in this project. Check:
   are typos realistic? Is the question_intent label actually correct
   for the phrasing? Does the JSON target match your actual schema field
   names exactly (not the reference version's simplified names)?
3. Decide on final dataset size - 1500 is a reasonable starting point
   for a narrow, bounded-schema task like this. You can always regenerate
   larger if val accuracy looks weak.

## Day 3: QLoRA fine-tuning

1. `pip install unsloth trl peft accelerate bitsandbytes --break-system-packages`
2. Run `train_qlora.py`. Watch `nvidia-smi` in another terminal the first
   time - if you see VRAM climbing toward the ceiling, reduce
   `per_device_train_batch_size` to 1 and increase
   `gradient_accumulation_steps` to compensate (same effective batch size,
   less peak memory).
3. Look at the val-set JSON-validity check the script prints at the end
   BEFORE looking at the loss curve. A pretty loss curve with broken JSON
   output is the same trap as the constant-label bug from Week 1 - don't
   repeat it here.
4. **Run the same val-set inputs through the BASE (non-fine-tuned) model
   too, and compare.** This before/after comparison IS your fine-tuning
   demonstration - it's the evidence that fine-tuning actually did
   something, not just a claim that you ran a script. Save both outputs
   side by side for your writeup/demo.
5. Export to GGUF (the script does this automatically at the end).

## Day 4: Integration + the encoder/CLS touchpoint

1. `ollama create parse-query-ft -f Modelfile.parse_query` (fix the FROM
   path first per the comment in that file).
2. Swap `parse_query`'s backend to this custom model the same way you did
   `generate_report` on Day 1, but pointing at `parse-query-ft` instead
   of the base model.
3. Re-run your 5 edge-case tests end-to-end through the full pipeline.
4. **Embedding model for Qdrant RAG** - this is where a real encoder/CLS
   model belongs in this project (see the conversation reasoning: CLS/MLM
   don't fit the generative nodes, but retrieval embeddings are a
   legitimate, correctly-motivated encoder use case). If your current
   retrieval isn't already using a proper sentence-embedding model:

   ```python
   from sentence_transformers import SentenceTransformer
   # a small, well-established encoder model - internally BERT-derived,
   # CLS-pooled, MLM-pretrained, then further tuned for embedding quality
   embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

   def embed_text(text: str) -> list[float]:
       return embedder.encode(text, normalize_embeddings=True).tolist()
   ```
   Re-embed your MoA text chunks with this and re-upsert into Qdrant if
   you're not already using a real embedding model for retrieval. This
   is a small, clean addition - don't over-scope it into a bigger
   redesign than it needs to be.

## Day 5: Documentation + buffer

Write up, for your README/report/viva prep, honestly and specifically
(not just concept names dropped in a list):

- **Which base model, and why** - Qwen2.5-7B-Instruct uses grouped-query
  attention (fewer KV heads than query heads, for inference efficiency)
  and RoPE for positional encoding. Explain what RoPE actually buys you
  (relative position encoding baked into the attention dot-product,
  better length generalization than learned absolute embeddings) - you
  inherited this by choosing the model, and that's an honest thing to
  say, not something to overclaim as "I implemented."
- **What you fine-tuned and why that specific node** - parse_query was
  tractable (bounded schema, synthetic data was feasible); generate_report
  was NOT fine-tuned, and say why (no gold-standard report dataset
  existed to train against - fabricating one would have been worse than
  not fine-tuning).
- **The LoRA config choices** - rank, alpha, which projection layers
  (q/k/v/o_proj are literally the multi-head attention projections -
  this is your concrete, defensible answer to "where does multi-head
  attention show up in what you built").
- **Before/after comparison numbers** from Day 3, honestly reported even
  if the improvement is modest - a small, honestly-reported delta is
  more credible than an inflated claim, consistent with how you handled
  the LOOCV/constant-label issues earlier in this project.
- **The CLS/embedding-model note** - where it's used (retrieval), why
  encoder-style pooling makes sense there specifically and not in the
  generative nodes.
- Point to your BERT fine-tuning / positional encoding / attention
  coursework as separate, already-completed proof of from-scratch
  understanding of the encoder-side concepts, rather than re-deriving
  it here.

Use remaining time as buffer - something in Days 3-4 is likely to need
debugging (GGUF export filenames, chat template mismatches, and VRAM
tuning are the most probable snags, in that order).
