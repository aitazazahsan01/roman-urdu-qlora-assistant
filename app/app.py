EXAMPLES = [
    ["Pakistan ka dar-ul-hukumat kya hai?", ""],
    ["Ek chota nazam likhein mausam ke baray mein.", ""],
]


def build_app(adapter_dir: str) -> gr.Blocks:
    try:
        assistant = RomanUrduQLoRAAssistant(adapter_dir)
        load_error = None
    except Exception as exc:  # no CUDA, adapter not trained yet, bad path, etc.
