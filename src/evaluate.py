        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU found. Real evaluation needs a GPU -- run on Kaggle, or pass --smoke-test.")
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=build_bnb_config(), device_map="auto")
        model = PeftModel.from_pretrained(base, str(args.adapter_dir))
        eval_ds = Dataset.load_from_disk(str(args.data_dir / "validation"))

    held_out = eval_ds.select(range(min(args.n_eval, len(eval_ds))))
    references = [row["output"] for row in held_out]
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "roman_urdu_qa"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", type=Path, default=None, help="Required unless --smoke-test")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        tokenizer.pad_token = tokenizer.eos_token

    if args.smoke_test:
        print("Running in --smoke-test mode: tiny random-weight model + untrained LoRA adapter, CPU.")
        config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        config.num_hidden_layers = 2
        config.hidden_size = 32
        config.intermediate_size = 64
    tuned_scores = rouge_l_summary(references, tuned_predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    all_results["base_zeroshot"] = base_scores
    all_results["qlora_tuned"] = tuned_scores
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    python src/evaluate.py --adapter-dir outputs/qwen3-8b-roman-urdu-qlora   # needs CUDA
    python src/evaluate.py --smoke-test
"""

import argparse
import json
from pathlib import Path

