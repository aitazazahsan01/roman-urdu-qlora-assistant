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

if __name__ == "__main__":
    main()
    return args


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
    p.add_argument("--n-eval", type=int, default=40, help="How many held-out rows to generate on")
    p.add_argument("--n-samples", type=int, default=8, help="How many to also save as full transcripts")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "metrics.json")
    p.add_argument("--samples-output", type=Path, default=ROOT / "results" / "sample_completions.json")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    if not args.smoke_test and args.adapter_dir is None:
        p.error("--adapter-dir is required unless --smoke-test")
        }
        for i in range(n_samples)
    ]
    with open(args.samples_output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} sample completions to {args.samples_output}")


                "instruction": ["Pakistan ka dar-ul-hukumat kya hai?", "Ek chota nazam likhein."],
                "input": ["", ""],
                "output": ["Islamabad Pakistan ka dar-ul-hukumat hai.", "Chaand raat mein chamakta hai."],
            }
        )
        args.output = args.output.parent / "smoke-test-metrics.json"
        args.samples_output = args.samples_output.parent / "smoke-test-sample_completions.json"
    else:

    print(f"Base (zero-shot)  ROUGE-L F: {base_scores['rougeL_fmeasure_mean']:.4f}  (n={base_scores['n']})")
    print(f"QLoRA-tuned        ROUGE-L F: {tuned_scores['rougeL_fmeasure_mean']:.4f}  (n={tuned_scores['n']})")
    print(f"Saved metrics to {args.output}")

    chart_path = args.output.parent / ("smoke-test-rouge_comparison.png" if args.smoke_test else "rouge_comparison.png")
    plot_rouge_comparison(base_scores, tuned_scores, chart_path)


    base_predictions, tuned_predictions = [], []
    for row in held_out:
        with model.disable_adapter():
            base_predictions.append(generate_completion(model, tokenizer, row["instruction"], row["input"]))
        tuned_predictions.append(generate_completion(model, tokenizer, row["instruction"], row["input"]))

    base_scores = rouge_l_summary(references, base_predictions)
        config.num_attention_heads = 2
        config.num_key_value_heads = 1
        base = AutoModelForCausalLM.from_config(config)
        from peft import get_peft_model

        model = get_peft_model(base, build_lora_config())  # untrained (LoRA B is zero-init) -- fine for a smoke test
        eval_ds = Dataset.from_dict(
            {
"""Evaluate a QLoRA adapter: generate on held-out prompts with the model both
with and without the adapter (peft's disable_adapter() context, so it's the
same loaded weights either way), score both arms with ROUGE-L, and save a
handful of full transcripts for qualitative reading -- a single scalar
undersells a small instruction-tuning result, so results/sample_completions.json
matters as much as results/metrics.json here.

Usage:
