        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU found. Real evaluation needs a GPU -- run on Kaggle, or pass --smoke-test.")
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=build_bnb_config(), device_map="auto")
        model = PeftModel.from_pretrained(base, str(args.adapter_dir))
        eval_ds = Dataset.load_from_disk(str(args.data_dir / "validation"))

    held_out = eval_ds.select(range(min(args.n_eval, len(eval_ds))))
    references = [row["output"] for row in held_out]
