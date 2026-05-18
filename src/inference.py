"""Single-prompt inference wrapper: load the 4-bit base model plus the LoRA
adapter (loaded dynamically via PeftModel, not merged -- merging a LoRA delta
into a 4-bit-quantized base isn't a clean operation, it needs dequantizing to
fp16 first, producing a second ~16GB artifact to manage. Dynamic loading keeps
the shareable artifact tiny (the adapter alone, tens of MB) and uses the exact
same quantization path at inference as at training). Used by app/app.py and
for quick manual checks.

Needs a CUDA GPU (bitsandbytes' 4-bit path is GPU-only) -- this will not run
on a CPU-only machine.

Usage:
    python src/inference.py --adapter-dir outputs/qwen3-8b-roman-urdu-qlora --text "..."
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qlora_utils import BASE_MODEL_NAME, build_bnb_config, generate_completion


class RomanUrduQLoRAAssistant:
    def __init__(self, adapter_dir_or_hub_id: str, base_model_name_or_path: str = BASE_MODEL_NAME, max_new_tokens: int = 256):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU found. This model needs a GPU to run the 4-bit base at usable speed "
                "(e.g. a Kaggle T4, or a GPU-tier Space) -- it will not run on a CPU-only machine."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, quantization_config=build_bnb_config(), device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir_or_hub_id)
        self.max_new_tokens = max_new_tokens

    def generate(self, instruction: str, input_text: str = "") -> str:
        return generate_completion(self.model, self.tokenizer, instruction, input_text, self.max_new_tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True, help="Local path or HF Hub repo id")
    p.add_argument("--text", required=True, help="The instruction")
    p.add_argument("--input", default="", help="Optional supplementary input")
    args = p.parse_args()

    assistant = RomanUrduQLoRAAssistant(args.adapter_dir)
    print(assistant.generate(args.text, args.input))


if __name__ == "__main__":
    main()
