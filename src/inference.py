            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, quantization_config=build_bnb_config(), device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir_or_hub_id)
        self.max_new_tokens = max_new_tokens

    def generate(self, instruction: str, input_text: str = "") -> str:
        return generate_completion(self.model, self.tokenizer, instruction, input_text, self.max_new_tokens)


if __name__ == "__main__":
    main()


class RomanUrduQLoRAAssistant:
    def __init__(self, adapter_dir_or_hub_id: str, base_model_name_or_path: str = BASE_MODEL_NAME, max_new_tokens: int = 256):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU found. This model needs a GPU to run the 4-bit base at usable speed "
                "(e.g. a Kaggle T4, or a GPU-tier Space) -- it will not run on a CPU-only machine."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        if self.tokenizer.pad_token is None:
