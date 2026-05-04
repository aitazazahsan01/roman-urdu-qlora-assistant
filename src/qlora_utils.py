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
