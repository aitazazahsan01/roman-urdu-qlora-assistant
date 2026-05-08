        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        fp16=use_4bit,  # only meaningful (and safe) when actually running on CUDA
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        gradient_checkpointing=use_4bit,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
    prepare_for_training,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "roman_urdu_qa"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", default=BASE_MODEL_NAME)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "qwen3-8b-roman-urdu-qlora")
    p.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
"""Fine-tune Qwen3-8B into a Roman Urdu instruction-following assistant via
4-bit QLoRA.

Real quantized QLoRA fine-tuning needs CUDA (bitsandbytes' 4-bit path is
GPU-only) -- run this for real on a Kaggle T4 via kaggle/train_kernel.ipynb
(this file's logic is inlined there) or any other CUDA machine.

--smoke-test runs a tiny random-weight Qwen3-architecture model in full
precision on CPU, against a real (but tiny) slice of the actual filtered
Roman-Urdu data, to validate the data pipeline, chat-template formatting, loss
masking, and LoRA adapter attachment before ever spending GPU quota. It does
NOT and cannot validate the real 4-bit bitsandbytes quantization path itself --
that can only be exercised on a real CUDA machine.

Usage:
    python src/train_qlora.py --smoke-test
    python src/train_qlora.py --output-dir outputs/qwen3-8b-roman-urdu-qlora   # needs CUDA
            "(kaggle/train_kernel.ipynb) or another CUDA machine, or pass --smoke-test to validate "
            "the pipeline on CPU instead."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.smoke_test:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.num_hidden_layers = 2
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_attention_heads = 2
        config.num_key_value_heads = 1
        model = AutoModelForCausalLM.from_config(config)
    else:
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from qlora_utils import (
    BASE_MODEL_NAME,
    MAX_SEQ_LENGTH,
    SFTDataCollator,
    build_bnb_config,
    format_and_mask,
    plot_training_loss,
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, quantization_config=build_bnb_config(), device_map="auto"
        )

    model = prepare_for_training(model, use_4bit=use_4bit)
    model.print_trainable_parameters()

    train_ds = load_split(args.data_dir, "train")
    eval_ds = load_split(args.data_dir, "validation")
    if args.smoke_test:
        train_ds = train_ds.select(range(min(16, len(train_ds))))
        eval_ds = eval_ds.select(range(min(8, len(eval_ds))))

    def tokenize(example):
        return format_and_mask(example["instruction"], example["input"], example["output"], tokenizer, args.max_seq_length)

    train_features = train_ds.map(tokenize, remove_columns=train_ds.column_names)
        eval_dataset=eval_features,
        processing_class=tokenizer,
        data_collator=SFTDataCollator(tokenizer),
    )
    trainer.train()

    loss_plot_name = "smoke-test-training_loss.png" if args.smoke_test else "training_loss.png"
    plot_training_loss(trainer.state.log_history, ROOT / "results" / loss_plot_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))  # adapter only (peft model)
    tokenizer.save_pretrained(str(args.output_dir))

    run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny random-weight model, CPU-friendly, real (but tiny) data slice, to validate the pipeline "
        "before spending GPU quota. Cannot exercise the real 4-bit quantization path.",
    )
    return p.parse_args()


def load_split(data_dir: Path, split: str) -> Dataset:
    return Dataset.load_from_disk(str(data_dir / split))


def main():
    args = parse_args()
