"""
Direct in-process inference for the QLoRA-fine-tuned parse_query model —
NOT served via Ollama/GGUF. See DECISIONS.md #7 and
finetuning/session_log/actions.log for why: converting the merged model
to GGUF for Ollama crashed the host machine's IDE three times via OOM
(the write phase of the conversion spiked memory faster than any
polling-based safety net could reliably catch). Loading the ALREADY-
MERGED model (finetuning/parse_query_gguf/, produced successfully by
training — only the subsequent GGUF export step was the problem) directly
via 4-bit quantization for inference is architecturally different: it's
read-heavy (memory-mapped, reclaimable pages) rather than write-heavy
(dirty pages needing flush), and was verified safe — two isolated test
runs, memory stable at ~25GB available throughout both, vs. the GGUF
export's crash within single-digit seconds of the write phase starting.

VRAM budget, not just RAM: this model needs ~5GB in 4-bit. generate_report
separately loads qwen2.5:14b-instruct via Ollama (~9GB). Together that
exceeds this machine's 12GB card. Since parse_query and generate_report
never run concurrently within one agent pass, this module loads the
model, generates, and explicitly frees GPU memory on every call rather
than keeping it resident — trading ~35-40s load latency per call for
never having two models fight over VRAM. Consistent with this project's
existing stance that backend pipeline steps don't need to feel fast
(generate_report already accepts ~90-130s for the same reason).
"""

import asyncio
import json

from app.agent.local_llm_prompts import PARSE_QUERY_FULL_SYSTEM_PROMPT
from app.agent.schemas import ParseResult
from app.agent.state import TokenUsage

MODEL_DIR = str((__import__("pathlib").Path(__file__).resolve().parents[2] / "finetuning" / "parse_query_gguf"))
# Display name for the UI — the actual load path (MODEL_DIR) is the
# merged QLoRA-fine-tuned checkpoint, not a bare base-model identifier,
# so this spells out what it actually is rather than showing a local
# filesystem path to the user.
MODEL_DISPLAY_NAME = "Qwen2.5-7B-Instruct (QLoRA fine-tuned, 4-bit)"
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 300

# The before/after comparison (finetuning/outputs/before_after_comparison.json)
# found a specific, bounded failure mode: malformed list-bracket syntax on
# the `comorbidities` field in ~20% (6/30) of val examples — reproduced
# live during Day-4 edge-case testing (see
# finetuning/session_log/actions.log). Not a truncation issue, not a
# schema-understanding issue (0 cases of valid-JSON-but-wrong-schema) —
# just an occasional formatting slip on one field. A bounded, known
# failure mode is exactly what a bounded retry is for.
#
# HONEST NOTE: retry is a mitigation, not a fix. Live testing (Day 4) hit
# a query ("Would lisinoprill work for a 60yo male, BP 150/95?") that
# failed on BOTH of 2 attempts, back to back — if failures were fully
# independent at a flat 20% rate, 2-in-a-row would be ~4% likely, so this
# could be ordinary bad luck, or this model may have query-specific weak
# spots that make it more (or less) than 20% for a given input. Bumped to
# 3 attempts as a cheap additional mitigation, not because the failure
# rate is precisely characterized — it isn't. Report this range honestly,
# don't imply retry makes the failure mode negligible. The graceful
# degradation path (parse_query.py catching FinetunedParseError and
# producing a clarification response) is what actually guarantees no
# crash, not the retry count.
MAX_ATTEMPTS = 3


class FinetunedParseError(Exception):
    """Raised when the fine-tuned model's output isn't valid JSON or
    doesn't validate against ParseResult after all retry attempts —
    mirrors app.agent.parse_query.ParseQueryOutputError for the Ollama
    path, kept as a distinct exception type since the failure modes/
    debugging steps differ (a bad load vs. a bad generation vs. a schema
    mismatch).

    Carries token_usage even on failure — 3 failed attempts is still real
    GPU compute spent, and hiding that from the UI would understate the
    actual cost of the known ~20% malformed-JSON failure mode."""

    def __init__(self, message: str, token_usage: TokenUsage | None = None):
        super().__init__(message)
        self.token_usage = token_usage


def _load_and_generate_sync(user_query: str) -> tuple[ParseResult, TokenUsage]:
    """Blocking, GPU-bound — must be called via asyncio.to_thread from
    the async node, never awaited directly. Loads the model ONCE, then
    attempts generation up to MAX_ATTEMPTS times against that single
    loaded model (retrying the ~40s load itself on a parse failure would
    double the cost for no reason — only generation needs retrying).
    Frees VRAM before returning or raising either way. Imports unsloth/
    torch lazily so importing this module doesn't pull in the whole ML
    stack for code paths that never call it (e.g. tests)."""
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    try:
        FastLanguageModel.for_inference(model)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": PARSE_QUERY_FULL_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_tokens_per_attempt = inputs["input_ids"].shape[1]
        output_tokens_so_far = 0

        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            output_tokens_so_far += new_tokens.shape[0]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            try:
                parsed = json.loads(decoded)
                result = ParseResult.model_validate(parsed)
                usage: TokenUsage = {
                    "model": MODEL_DISPLAY_NAME,
                    # The same prompt is re-sent on every retry attempt —
                    # summed across attempts for an honest total compute
                    # cost, not just the final successful call's cost.
                    "input_tokens": input_tokens_per_attempt * attempt,
                    "output_tokens": output_tokens_so_far,
                    "total_tokens": input_tokens_per_attempt * attempt + output_tokens_so_far,
                    "attempts": attempt,
                }
                return result, usage
            except (json.JSONDecodeError, Exception) as e:
                last_error = f"attempt {attempt}/{MAX_ATTEMPTS}: {e}\nRaw: {decoded[:300]}"
                continue
        usage = {
            "model": MODEL_DISPLAY_NAME,
            "input_tokens": input_tokens_per_attempt * MAX_ATTEMPTS,
            "output_tokens": output_tokens_so_far,
            "total_tokens": input_tokens_per_attempt * MAX_ATTEMPTS + output_tokens_so_far,
            "attempts": MAX_ATTEMPTS,
        }
        raise FinetunedParseError(
            f"Fine-tuned model output failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}",
            token_usage=usage,
        )
    finally:
        # Explicit free — see module docstring on why this isn't kept
        # resident. Deleting the Python references alone isn't enough;
        # empty_cache() is what actually returns VRAM to the driver.
        del model, tokenizer
        torch.cuda.empty_cache()


async def parse_with_finetuned_model(user_query: str) -> tuple[ParseResult, TokenUsage]:
    """Async entry point for app.agent.parse_query. Runs the blocking
    load+generate(+retry) in a thread so it doesn't block the event loop.
    Parsing/validation happens the same way train_qlora.py's
    run_eval_pass did (JSON parse, then real-schema validation) — serving
    must match how this model was evaluated, not diverge from it."""
    return await asyncio.to_thread(_load_and_generate_sync, user_query)
