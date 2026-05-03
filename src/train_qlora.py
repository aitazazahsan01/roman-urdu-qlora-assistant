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
