"""
QLoRA fine-tune of a small instruct model on the parse_query task, using
Unsloth for memory-efficient training on a single consumer GPU.

Run on YOUR machine (RTX 5070 Ti, 12GB VRAM):
    pip install unsloth trl peft accelerate bitsandbytes --break-system-packages
    python train_qlora.py

Expect this to take somewhere in the range of 20-60 minutes for 1500
examples / 3 epochs on a 7-8B model in 4-bit on a 12GB card - exact time
depends on sequence length and your specific GPU. Watch nvidia-smi in
another terminal the first time to confirm you're not swapping/OOMing.

MANDATORY before/after comparison (continuation brief: "A fine-tune
without a documented before/after comparison is not evidence it worked"):
this script now runs the SAME val-set prompts through the BASE model
(before any LoRA is attached) and the FINE-TUNED model (after training),
and writes both outputs side by side to
outputs/before_after_comparison.json. Do not report a fine-tuning result
without this file - a pretty loss curve is not evidence on its own, see
DECISIONS.md #4's constant-label bug for why "it ran and the metric
looked fine" was already an expensive lesson once in this project.
"""

import json
import sys
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent.schemas import ParseResult  # noqa: E402 — path insert must come first

N_EVAL_EXAMPLES = 30  # spot-check, not the full val set — matches the original script's scope

# ---- config - these are the "parameter tuning" knobs worth explaining
# in your writeup, not just accepting as magic numbers ----
#
# Hardware assumed: RTX 5070 Ti (12GB VRAM), 32GB system RAM. Kept at
# 7B deliberately - this is the one fine-tuned node on a tight 3-5 day
# timeline, and QLoRA OOM/instability debugging on a bigger model is
# exactly the kind of thing that eats a day you don't have. Your
# hardware CAN go bigger (see the stretch-option block below) if you
# want to take that risk on purpose, not by default.
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # pre-quantized, faster download
MAX_SEQ_LENGTH = 1024  # our examples are short (patient description + JSON) - no need for 2k+
LORA_RANK = 16  # dimensionality of the LoRA update matrices - higher = more capacity to adapt, more VRAM/slower. 16 is a reasonable default for a narrow task like this, not a wide one.
LORA_ALPHA = 16  # scaling factor for the LoRA update, conventionally set equal to rank as a starting point
LORA_DROPOUT = 0  # Unsloth's fast path is optimized for dropout=0; add some (e.g. 0.05) if you see overfitting in val loss
LEARNING_RATE = 2e-4  # standard starting point for LoRA (much higher than full fine-tuning LR, since only a small param subset is being updated)
NUM_EPOCHS = 3
OUTPUT_DIR = "outputs"
GGUF_OUTPUT_DIR = "parse_query_gguf"

# ---- STRETCH OPTION: 13-14B on your hardware, if you want the risk ----
# BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
# Also change in TrainingArguments below:
#   optim = "paged_adamw_8bit"          # was "adamw_8bit" - pages optimizer
#                                          state to system RAM when VRAM is
#                                          tight; your 32GB RAM is what makes
#                                          this viable rather than OOMing
#   per_device_train_batch_size = 1     # was 2
#   gradient_accumulation_steps = 8     # was 4 - keeps effective batch size ~similar
# Expect this to be meaningfully slower and more fragile to tune than the
# 7B default. Only worth it if the 7B fine-tune's before/after comparison
# (Day 3 of the plan) looks weak and you have a spare day to spend on it.


def load_jsonl_as_dataset(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def run_eval_pass(model, tokenizer, val_examples: list[dict], label: str) -> dict:
    """Run val_examples through `model` and score each output three ways,
    weakest to strongest claim: (1) valid JSON at all, (2) validates
    against the REAL ParseResult schema (field names/types/nesting all
    correct — this is the check that would have caught the Step 0
    reference-schema mismatch if it had shipped untested), (3) exact-match
    on the fields that should be unambiguous given the prompt (compound
    raw_names verbatim including typos, question_intent, patient.age,
    patient.sex) - this is the real "did fine-tuning teach it the task"
    signal, not just "did it produce syntactically valid output."

    Called once on the BASE model (before LoRA) and once on the
    FINE-TUNED model (after training) with the identical val_examples, so
    the two results are directly comparable - see module docstring."""
    FastLanguageModel.for_inference(model)
    results = []
    for ex in val_examples:
        expected = json.loads(ex["messages"][2]["content"])
        prompt = tokenizer.apply_chat_template(
            ex["messages"][:2], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=300, use_cache=True)
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        row = {"user_query": ex["messages"][1]["content"], "expected": expected, "generated_raw": decoded,
               "valid_json": False, "schema_valid": False, "exact_field_match": False}

        try:
            parsed = json.loads(decoded)
            row["valid_json"] = True
        except json.JSONDecodeError:
            results.append(row)
            continue

        try:
            validated = ParseResult.model_validate(parsed)
            row["schema_valid"] = True
        except Exception as e:
            row["schema_error"] = str(e)[:300]
            results.append(row)
            continue

        expected_raw_names = sorted(c["raw_name"] for c in expected["compounds"])
        got_raw_names = sorted(c.raw_name for c in validated.compounds)
        expected_patient = expected["parsed_patient"]["patient"]
        row["exact_field_match"] = (
            expected_raw_names == got_raw_names
            and expected["question_intent"] == validated.question_intent
            and expected_patient["age"] == validated.parsed_patient.patient.age
            and expected_patient["sex"] == validated.parsed_patient.patient.sex.value
        )
        results.append(row)

    n = len(results)
    summary = {
        "label": label,
        "n": n,
        "valid_json_rate": sum(r["valid_json"] for r in results) / n,
        "schema_valid_rate": sum(r["schema_valid"] for r in results) / n,
        "exact_field_match_rate": sum(r["exact_field_match"] for r in results) / n,
    }
    print(f"\n[{label}] valid_json={summary['valid_json_rate']:.0%} "
          f"schema_valid={summary['schema_valid_rate']:.0%} "
          f"exact_field_match={summary['exact_field_match_rate']:.0%}")
    return {"summary": summary, "results": results}


def formatting_func(example, tokenizer):
    """Apply the model's chat template to our system/user/assistant
    messages. This matters: the fine-tuned model needs to see EXACTLY
    the same chat template format it'll be served with at inference
    time via Ollama, or the fine-tuning won't transfer cleanly."""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # auto-detect bf16/fp16 support
        load_in_4bit=True,
    )

    print("Loading datasets...")
    train_ds = load_jsonl_as_dataset("data/train.jsonl")
    val_ds = load_jsonl_as_dataset("data/val.jsonl")
    val_examples = [json.loads(line) for line in open("data/val.jsonl")][:N_EVAL_EXAMPLES]

    # ---- BASE model eval, BEFORE any LoRA is attached ----
    # This is the "before" half of the mandatory before/after comparison
    # (continuation brief). Must happen here, before get_peft_model() -
    # once LoRA is attached the base weights are still technically
    # underneath, but Unsloth's for_inference/PEFT wrapper means "the
    # model" from here on IS the adapted model; there's no clean way to
    # ask it to temporarily ignore its own adapter mid-script, so capture
    # the true base behavior now while it's unambiguous.
    print("\n=== Evaluating BASE model (before fine-tuning) ===")
    base_eval = run_eval_pass(model, tokenizer, val_examples, label="base_model")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],  # standard attention + MLP projection layers - this IS the
            # "which parts of multi-head attention am I adapting" answer
            # for your writeup: q/k/v/o_proj are literally the query,
            # key, value, and output projections of multi-head attention.
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's memory-optimized checkpointing - this is what makes 7B fine-tuning fit in 12GB
        random_state=3407,
    )

    train_ds = train_ds.map(lambda ex: formatting_func(ex, tokenizer))
    val_ds = val_ds.map(lambda ex: formatting_func(ex, tokenizer))

    # NOTE: no eval_dataset / eval_strategy on the trainer itself. First
    # run on this machine (RTX 5070 Ti, very new Blackwell/sm_120 silicon,
    # FA2/xformers both unavailable -> attention falls back to an
    # unoptimized path per the startup banner) genuinely HUNG right after
    # HF Trainer's automatic step-50 internal eval pass tried to hand
    # control back to training - confirmed a real stall, not just slow:
    # three checks several minutes apart showed byte-identical log output
    # AND byte-identical /proc/<pid>/io counters, with the GPU sitting at
    # 0% utilization / ~4W the whole time. Killed and removed the
    # in-training eval loop rather than chase a driver/kernel-level
    # hang on bleeding-edge hardware during a 3-5 day window - the
    # mandatory before/after comparison (run_eval_pass, above/below) is
    # a separate, self-contained eval pass that doesn't interact with
    # the trainer's internal state the same way, so it isn't suspected
    # of causing this.
    #
    # SECOND real bug found running this on the actual hardware: with
    # default save_strategy, HF Trainer's automatic checkpoint save at
    # step 500 crashed with `PicklingError: Can't pickle <class
    # 'trl.trainer.sft_config.SFTConfig'>: it's not the same object as
    # trl.trainer.sft_config.SFTConfig` - a module-identity mismatch,
    # almost certainly from Unsloth's compiled-cache patching
    # (finetuning/unsloth_compiled_cache/UnslothSFTTrainer.py) loading a
    # second, non-identical copy of the same class. This happened AFTER
    # training itself finished converging (loss 1.807 -> 0.058 over
    # ~500/507 steps) - the crash is in the checkpoint-SAVE code path,
    # not the training computation. Fix: save_strategy="no". This script
    # doesn't need intermediate checkpoints anyway - it exports the
    # final trained model straight to GGUF after training completes.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # effective batch size 8
            warmup_steps=10,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="no",
            optim="adamw_8bit",  # 8-bit optimizer state - another memory saver, needed to fit in 12GB alongside the 4-bit model
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training complete. Final stats: {trainer_stats}")

    # ---- honest evaluation before you trust this for anything ----
    # Don't just report the training loss curve looking nice - that's
    # the same trap as the constant-label side-effect model from Week 1.
    # Run the SAME val_examples used for the base-model pass above, so
    # the two are directly comparable - this IS the fine-tuning
    # demonstration, not the loss curve.
    print("\n=== Evaluating FINE-TUNED model (after training) ===")
    finetuned_eval = run_eval_pass(model, tokenizer, val_examples, label="finetuned_model")

    print("\n=== Before / after comparison ===")
    print(f"valid_json_rate:        base={base_eval['summary']['valid_json_rate']:.0%}  "
          f"finetuned={finetuned_eval['summary']['valid_json_rate']:.0%}")
    print(f"schema_valid_rate:      base={base_eval['summary']['schema_valid_rate']:.0%}  "
          f"finetuned={finetuned_eval['summary']['schema_valid_rate']:.0%}")
    print(f"exact_field_match_rate: base={base_eval['summary']['exact_field_match_rate']:.0%}  "
          f"finetuned={finetuned_eval['summary']['exact_field_match_rate']:.0%}")
    if finetuned_eval["summary"]["schema_valid_rate"] < 0.8:
        print("\nWARNING: schema_valid_rate isn't close to 100% - something needs fixing before")
        print("trusting this model (more epochs, a cleaner chat template match, or a lower")
        print("learning rate are the usual culprits). Report this honestly either way.")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    comparison_path = Path(OUTPUT_DIR) / "before_after_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump({"base_model": base_eval, "finetuned_model": finetuned_eval}, f, indent=2)
    print(f"\nWrote full before/after comparison (all {N_EVAL_EXAMPLES} examples, both models) to {comparison_path}")
    print("This file is the actual evidence for the fine-tuning claim - reference it directly")
    print("in the writeup, don't just cite the summary numbers from memory.")

    # ---- export for Ollama ----
    print(f"\nExporting merged model to GGUF (q4_k_m) at {GGUF_OUTPUT_DIR}/...")
    model.save_pretrained_gguf(GGUF_OUTPUT_DIR, tokenizer, quantization_method="q4_k_m")
    print("Done. Next: write the Ollama Modelfile pointing at the .gguf file (see Modelfile in this dir)")


if __name__ == "__main__":
    main()
