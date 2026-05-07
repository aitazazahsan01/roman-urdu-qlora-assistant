EXAMPLES = [
    ["Pakistan ka dar-ul-hukumat kya hai?", ""],
    ["Ek chota nazam likhein mausam ke baray mein.", ""],
]


def build_app(adapter_dir: str) -> gr.Blocks:
    try:
        assistant = RomanUrduQLoRAAssistant(adapter_dir)
        load_error = None
    except Exception as exc:  # no CUDA, adapter not trained yet, bad path, etc.
        gr.Markdown(
            "# Roman Urdu QLoRA Assistant\n"
            "`Qwen/Qwen3-8B` fine-tuned via 4-bit QLoRA on a small Roman Urdu instruction "
            "set. This is style/behavior adaptation on ~485 examples, not knowledge "
            "injection -- expect fluent Roman Urdu register, not encyclopedic accuracy."
        )
        with gr.Row():
            with gr.Column():
                instruction_box = gr.Textbox(label="Instruction", lines=3, placeholder="Apna sawaal yahan likhein...")
                input_box = gr.Textbox(label="Extra context (optional)", lines=2)
                submit_btn = gr.Button("Generate", variant="primary")
        assistant = None
        load_error = str(exc)

    def respond(instruction: str, input_text: str):
        if load_error:
            return f"Model failed to load: {load_error}"
        if not instruction.strip():
            return "Type an instruction first."
        return assistant.generate(instruction, input_text)

    with gr.Blocks(title="Roman Urdu QLoRA Assistant") as demo:
