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
