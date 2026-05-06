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
