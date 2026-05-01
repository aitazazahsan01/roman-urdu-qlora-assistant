        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return get_peft_model(model, build_lora_config())
