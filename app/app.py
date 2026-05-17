"""Gradio demo: type an instruction (optionally with extra context) and get a
Roman Urdu response from the QLoRA-tuned assistant.

Needs a CUDA GPU (bitsandbytes' 4-bit path is GPU-only) -- this will NOT run
on the local CPU-only dev machine. Run it on a GPU-tier Space, a Kaggle
notebook with a Gradio cell, or any other CUDA machine.

Usage:
    python app/app.py --adapter-dir outputs/qwen3-8b-roman-urdu-qlora
"""

import argparse
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import RomanUrduQLoRAAssistant  # noqa: E402

EXAMPLES = [
    ["Pakistan ka dar-ul-hukumat kya hai?", ""],
    ["Ek chota nazam likhein mausam ke baray mein.", ""],
]


def build_app(adapter_dir: str) -> gr.Blocks:
    try:
        assistant = RomanUrduQLoRAAssistant(adapter_dir)
        load_error = None
    except Exception as exc:  # no CUDA, adapter not trained yet, bad path, etc.
        assistant = None
        load_error = str(exc)

    def respond(instruction: str, input_text: str):
        if load_error:
            return f"Model failed to load: {load_error}"
        if not instruction.strip():
            return "Type an instruction first."
        return assistant.generate(instruction, input_text)

    with gr.Blocks(title="Roman Urdu QLoRA Assistant") as demo:
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
                gr.Examples(examples=EXAMPLES, inputs=[instruction_box, input_box])
            with gr.Column():
                output_box = gr.Textbox(label="Response", lines=8)

        submit_btn.click(respond, inputs=[instruction_box, input_box], outputs=[output_box])

    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", default=str(ROOT / "outputs" / "qwen3-8b-roman-urdu-qlora"))
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    demo = build_app(args.adapter_dir)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
