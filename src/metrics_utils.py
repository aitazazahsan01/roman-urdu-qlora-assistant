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
