"""
ISOLATED test: load the already-merged fine-tuned model
(finetuning/parse_query_gguf/, on disk, safe, from the successful training
run) directly via Unsloth's 4-bit loading and run ONE generation call.

Deliberately standalone (not wired into app/agent/ yet) so this can be
run under close memory monitoring before trusting it inside the actual
agent pipeline. See finetuning/session_log/actions.log for why: the
GGUF-conversion path crashed the system 3 times; this is a different
approach (read-heavy, in-VRAM quantized load) that should not carry the
same risk, but "should not" needs to be verified, not assumed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent.local_llm_prompts import PARSE_QUERY_FULL_SYSTEM_PROMPT  # noqa: E402

from unsloth import FastLanguageModel

MODEL_DIR = "parse_query_gguf"  # the already-merged model, run from finetuning/
MAX_SEQ_LENGTH = 1024

# FIRST TEST RUN BUG (caught by testing, not review): used a simplified
# placeholder prompt here instead of the real training prompt, and got
# structurally wrong output as a direct result (parsed_patient nested
# INSIDE a compound entry, wrong field names, most fields missing) - a
# live demonstration of exactly the train/serve prompt-mismatch risk
# already documented in local_llm_prompts.py's own docstring. Fixed by
# importing the actual prompt instead of re-describing it by hand.
SYSTEM_PROMPT = PARSE_QUERY_FULL_SYSTEM_PROMPT
TEST_QUERY = "How effective would this be? Evaluating losartan 50mg for a 60-year-old male, BP 150/95."


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading model from {MODEL_DIR} in 4-bit...")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded in {time.time() - t0:.1f}s")

    FastLanguageModel.for_inference(model)

    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": TEST_QUERY}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"[{time.strftime('%H:%M:%S')}] Generating...")
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=300, use_cache=True)
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[{time.strftime('%H:%M:%S')}] Generated in {time.time() - t0:.1f}s")

    print("\n--- OUTPUT ---")
    print(decoded)
    print("--- END ---")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
