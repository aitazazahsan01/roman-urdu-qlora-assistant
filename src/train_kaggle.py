"""End-to-end GPU driver: data prep + 4-bit QLoRA fine-tuning + base-vs-tuned
evaluation, in a single run. Meant for a Kaggle kernel (free T4 GPU) -- see
kaggle/kernel-metadata.json and scripts/build_kaggle_notebook.py, which
inlines this file (plus qlora_utils.py, data_prep.py, and metrics_utils.py)
into a self-contained, cell-by-cell notebook meant to be run and watched, not
just executed blind.

Written as flat top-level script code (no main() wrapper) so it can be dropped
in as notebook cells as-is. The `# ---- section ----` comments below are not
just visual dividers -- scripts/build_kaggle_notebook.py splits on them to
produce one notebook cell per stage. Also runs standalone:
`python src/train_kaggle.py` on any machine with a CUDA GPU and
requirements.txt installed.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from data_prep import build_splits
from metrics_utils import plot_rouge_comparison, rouge_l_summary
from qlora_utils import (
    BASE_MODEL_NAME,
    MAX_SEQ_LENGTH,
    SFTDataCollator,
    build_bnb_config,
    format_and_mask,
    generate_completion,
    plot_training_loss,
    prepare_for_training,
)

# ---- config ----
VAL_FRACTION = 0.1
SEED = 42
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
N_SAMPLE_COMPLETIONS = 8

ON_KAGGLE = Path("/kaggle/working").exists()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(__file__).resolve().parent.parent
ADAPTER_DIR = OUTPUT_ROOT / "outputs" / "qwen3-8b-roman-urdu-qlora"
RESULTS_PATH = OUTPUT_ROOT / "results" / "metrics.json"
SAMPLES_PATH = OUTPUT_ROOT / "results" / "sample_completions.json"

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU found. This script needs a real GPU (Kaggle: Settings -> Accelerator -> GPU T4 x1).")

set_seed(SEED)
print(f"Device: {torch.cuda.get_device_name(0)}  |  Output root: {OUTPUT_ROOT}")

# ---- load & quantize base model ----
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, quantization_config=build_bnb_config(), device_map="auto"
)

# ---- configure LoRA ----
model = prepare_for_training(base_model, use_4bit=True)
model.print_trainable_parameters()

# ---- load & format data ----
train_ds, eval_ds = build_splits(VAL_FRACTION, SEED)
print(f"train={len(train_ds)}  validation={len(eval_ds)}")


def tokenize(example):
    return format_and_mask(example["instruction"], example["input"], example["output"], tokenizer, MAX_SEQ_LENGTH)


train_features = train_ds.map(tokenize, remove_columns=train_ds.column_names)
eval_features = eval_ds.map(tokenize, remove_columns=eval_ds.column_names)

# ---- train ----
steps_per_epoch = -(-len(train_features) // (TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS))
total_steps = int(steps_per_epoch * NUM_EPOCHS)
warmup_steps = int(total_steps * 0.05)

training_args = TrainingArguments(
    output_dir=str(ADAPTER_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    fp16=True,  # not bf16 -- T4 is Turing (cc 7.5), no Ampere+ bf16 tensor cores
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    logging_steps=10,
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_features,
    eval_dataset=eval_features,
    processing_class=tokenizer,
    data_collator=SFTDataCollator(tokenizer),
)
trainer.train()

# ---- plot training loss ----
plot_training_loss(trainer.state.log_history, OUTPUT_ROOT / "results" / "training_loss.png")

# ---- save adapter ----
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print(f"Saved LoRA adapter to {ADAPTER_DIR}")

# ---- evaluate: base vs tuned + ROUGE-L ----
# Generate both arms from the SAME loaded model object via peft's
# disable_adapter() context manager, rather than loading the 8B base a second
# time -- a second full copy would not fit alongside the first in a T4's 16GB.
held_out = eval_ds.select(range(min(len(eval_ds), 40)))  # keep eval generation time reasonable
references = [row["output"] for row in held_out]

base_predictions, tuned_predictions = [], []
for row in held_out:
    with model.disable_adapter():
        base_predictions.append(generate_completion(model, tokenizer, row["instruction"], row["input"]))
    tuned_predictions.append(generate_completion(model, tokenizer, row["instruction"], row["input"]))

base_scores = rouge_l_summary(references, base_predictions)
tuned_scores = rouge_l_summary(references, tuned_predictions)
print(f"Base (zero-shot)  ROUGE-L F: {base_scores['rougeL_fmeasure_mean']:.4f}")
print(f"QLoRA-tuned        ROUGE-L F: {tuned_scores['rougeL_fmeasure_mean']:.4f}")

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
all_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
all_results["base_zeroshot"] = base_scores
all_results["qlora_tuned"] = tuned_scores
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved metrics to {RESULTS_PATH}")

# ---- plot rouge comparison ----
plot_rouge_comparison(base_scores, tuned_scores, OUTPUT_ROOT / "results" / "rouge_comparison.png")

# ---- sample completions ----
samples = []
for row, base_pred, tuned_pred in zip(
    held_out.select(range(min(N_SAMPLE_COMPLETIONS, len(held_out)))),
    base_predictions[:N_SAMPLE_COMPLETIONS],
    tuned_predictions[:N_SAMPLE_COMPLETIONS],
):
    samples.append(
        {
            "instruction": row["instruction"],
            "input": row["input"],
            "reference": row["output"],
            "base_zeroshot": base_pred,
            "qlora_tuned": tuned_pred,
        }
    )
    print(f"\nInstruction: {row['instruction']}")
    print(f"Reference:   {row['output']}")
    print(f"Base:        {base_pred}")
    print(f"QLoRA-tuned: {tuned_pred}")

with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2)
print(f"\nSaved {len(samples)} sample completions to {SAMPLES_PATH}")
