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
