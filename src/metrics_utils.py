    if _is_notebook():
        plt.show()
    plt.close()
    print(f"Saved ROUGE-L comparison chart to {output_path}")
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar([i - width / 2 for i in x], base_vals, width, label="Base (zero-shot)")
    plt.bar([i + width / 2 for i in x], tuned_vals, width, label="QLoRA-tuned")
    plt.xticks(list(x), labels)
    plt.ylabel("ROUGE-L")
    plt.title("Base vs QLoRA-tuned: ROUGE-L on held-out prompts")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


import matplotlib  # noqa: E402

# See qlora_utils._is_notebook's comment: force the non-interactive Agg
# backend outside a notebook kernel so plt.show() can never block on a GUI
# window that nothing will close in a headless script run.
if not _is_notebook():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rouge_l_summary(references: list, predictions: list) -> dict:
    """use_stemmer=False is deliberate: the default English Porter stemmer
    would silently mangle Roman-Urdu tokens if left on -- the same
    "don't apply English-tuned NLP defaults to Roman Urdu" discipline used
    elsewhere in this project (the tokenizer/chat-template choices)."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    precisions, recalls, fmeasures = [], [], []
    }


def plot_rouge_comparison(base_scores: dict, tuned_scores: dict, output_path: Path) -> None:
    """Grouped bar chart: base (zero-shot) vs QLoRA-tuned, precision/recall/F.
    Saves a PNG and also renders inline when run inside a notebook."""
    labels = ["Precision", "Recall", "F-measure"]
    keys = ["rougeL_precision_mean", "rougeL_recall_mean", "rougeL_fmeasure_mean"]
    base_vals = [base_scores[k] for k in keys]
    tuned_vals = [tuned_scores[k] for k in keys]

    x = range(len(labels))
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)["rougeL"]
        precisions.append(score.precision)
        recalls.append(score.recall)
        fmeasures.append(score.fmeasure)

    n = len(fmeasures)
    return {
        "n": n,
        "rougeL_precision_mean": sum(precisions) / n if n else 0.0,
        "rougeL_recall_mean": sum(recalls) / n if n else 0.0,
        "rougeL_fmeasure_mean": sum(fmeasures) / n if n else 0.0,
"""ROUGE-L scoring, shared by evaluate.py and the Kaggle notebook's inline
base-vs-tuned comparison, so both arms are scored identically.
"""

from pathlib import Path

from rouge_score import rouge_scorer


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
