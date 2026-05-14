        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return get_peft_model(model, build_lora_config())

    repetition_penalty/no_repeat_ngram_size default to non-trivial values, not
    1.0/0 -- a first real Kaggle run (485-row fine-tune, do_sample=False) showed
    several completions degrade into verbatim repetition loops on longer
    outputs. Greedy decoding alone has no mechanism to break out of one once
    the model locks onto it; these two args do. See README "Results" for the
    actual examples that motivated this."""
    messages = build_messages(instruction, input_text)
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
    )["input_ids"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
def format_and_mask(instruction: str, input_text: str, output: str, tokenizer, max_length: int = MAX_SEQ_LENGTH) -> dict:
    """Tokenize one (instruction, input, output) example into the model's chat
    format, with the prompt portion masked out of the loss (-100) so the model
    is only trained to predict the assistant's response, not the user turn."""
    messages = build_messages(instruction, input_text)

    # apply_chat_template(tokenize=True) returns a BatchEncoding (dict-like,
    # {"input_ids": [...], "attention_mask": [...]}) in this transformers
    # version, not a plain token-id list -- always index ["input_ids"]
    # explicitly rather than treating the result as a list.
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": output}], tokenize=True, add_generation_prompt=False
    )["input_ids"]
"""Shared QLoRA plumbing: quantization config, LoRA config, chat-template
formatting with prompt-token loss-masking, and the SFT data collator.

Used by both train_qlora.py (local) and train_kaggle.py (Kaggle GPU); inlined
into kaggle/train_kernel.ipynb by scripts/build_kaggle_notebook.py so the two
never drift.

Qwen3's chat template always wraps assistant content in a <think>...</think>
block. With enable_thinking=False the block is forced empty at generation time
-- verified (by tokenizing real examples both ways and diffing token ids) that
building a full [user, assistant] conversation with add_generation_prompt=False
produces that exact same empty-think prefix, so the prompt built here with
add_generation_prompt=True is byte-identical to the prefix of the full labeled
sequence. That prefix-identity is what format_and_mask relies on to mask
exactly the prompt tokens out of the loss; the assertion below fails loudly
(rather than silently mis-masking) if a future transformers/Qwen3 chat-template
    both a causal-LM label sequence and a prompt-masked region at once)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list) -> dict:
        labels = [f["labels"] for f in features]
        no_labels = [{"input_ids": f["input_ids"]} for f in features]

        batch = self.tokenizer.pad(no_labels, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]

        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded_labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)

revision breaks this property.
"""

from pathlib import Path

import torch
from transformers import BitsAndBytesConfig


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
    """4-bit NF4 quantization config. compute_dtype is float16, NOT bfloat16:
    a Kaggle T4 is a Turing GPU (compute capability 7.5) -- bf16 tensor-core
    support only starts at Ampere (cc 8.0). Most QLoRA recipes default to bf16
    because they assume A100-class hardware; using it on a T4 either silently
    falls back to slow emulated math or produces dtype mismatches against the
    Trainer's fp16 mixed-precision mode. Must pair with TrainingArguments(fp16=True)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def build_lora_config():
    from peft import LoraConfig
        plt.show()
    plt.close()
    print(f"Saved loss curve to {output_path}")


def prepare_for_training(model, use_4bit: bool):
    """Attach LoRA adapters. When use_4bit, first runs peft's
    prepare_model_for_kbit_training -- the standard, necessary glue step when
    combining 4-bit loading + gradient checkpointing (enables requires_grad on
    the input embeddings and casts norms to fp32); skipping it is a well-known
    QLoRA gotcha (`element 0 of tensors does not require grad`). Not needed for
    the full-precision smoke-test model, which is tiny enough to train without
    gradient checkpointing at all."""
    from peft import get_peft_model, prepare_model_for_kbit_training

    if use_4bit:
            do_sample=False,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    completion_ids = output_ids[0][prompt_ids.shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def plot_training_loss(log_history: list, output_path: Path) -> None:
    """Plots train/eval loss vs. step from Trainer.state.log_history. Saves a
    PNG and also renders inline (plt.show()) when run inside a notebook."""
    train_steps = [e["step"] for e in log_history if "loss" in e and "eval_loss" not in e]
    train_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_steps = [e["step"] for e in log_history if "eval_loss" in e]

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


def build_messages(instruction: str, input_text: str) -> list:
    user = instruction if not input_text.strip() else f"{instruction}\n\n{input_text}"
    return [{"role": "user", "content": user}]


