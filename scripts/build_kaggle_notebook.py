"""Generate kaggle/train_kernel.ipynb by inlining src/qlora_utils.py,
src/metrics_utils.py, src/data_prep.py, and src/train_kaggle.py into a single
self-contained notebook.

Unlike the terser notebooks in the sibling projects (which the agent drove
end-to-end via the Kaggle API), THIS notebook is meant to be run and watched
by a human, cell by cell -- so train_kaggle.py's `# ---- section ----` comments
aren't just visual dividers, they're split points: each becomes its own
notebook cell with an explanatory markdown cell in front of it. Still zero
hand-duplicated pipeline code -- only the markdown prose and the pip-install
line are notebook-only, everything else is inlined straight from src/.

`kaggle kernels push` only pushes one code file, so rather than hand-maintaining
a separate copy of the training pipeline for Kaggle (and letting it drift from
the real src/ logic), this script assembles the notebook from the real source
files. Re-run it after changing any of the four files above.

Usage:
    python scripts/build_kaggle_notebook.py
"""

import json
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
OUT = Path(__file__).resolve().parent.parent / "kaggle" / "train_kernel.ipynb"


def read(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def strip_module_docstring(code: str) -> str:
    return re.sub(r'^"""[\s\S]*?"""\n+', "", code, count=1)


def strip_local_imports(code: str) -> str:
    """Strips both single-line (`from X import a, b`) and multi-line
    parenthesized (`from X import (\n    a,\n    b,\n)`) local imports --
    train_kaggle.py's `from qlora_utils import (...)` uses the latter."""
    code = re.sub(r"^from (qlora_utils|metrics_utils|data_prep) import \([\s\S]*?\)\n", "", code, flags=re.MULTILINE)
    code = re.sub(r"^from (qlora_utils|metrics_utils|data_prep) import .*\n", "", code, flags=re.MULTILINE)
    return code


def strip_main_block(code: str) -> str:
    """data_prep.py puts def main() and the __main__ guard last; the notebook
    doesn't need either (build_splits() is called directly in train_kaggle.py)."""
    return re.sub(r"\ndef main\(\)[\s\S]*\Z", "\n", code)


def strip_default_out_dir(code: str) -> str:
    """DEFAULT_OUT_DIR relies on __file__, which isn't defined when a cell
    runs inside a notebook kernel (unlike train_kaggle.py's OUTPUT_ROOT, this
    one isn't behind a ON_KAGGLE ternary, so it crashes unconditionally). It's
    only ever used as an argparse default inside main(), already stripped
    above -- safe, and necessary, to drop for the notebook."""
    return re.sub(r"^DEFAULT_OUT_DIR = .*\n", "", code, flags=re.MULTILINE)


def split_by_section_markers(code: str) -> list:
    """Splits train_kaggle.py on lines matching `# ---- name ----`, returning
    [(section_name, code_block), ...]. Code before the first marker (just the
    import statements) becomes its own leading 'imports' section."""
    pattern = re.compile(r"^# ---- (.+) ----$", re.MULTILINE)
    matches = list(pattern.finditer(code))
    sections = []
    if matches and matches[0].start() > 0:
        head = code[: matches[0].start()].strip()
        if head:
            sections.append(("imports", head))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
        block = code[start:end].strip()
        sections.append((m.group(1), block))
    return sections


SECTION_MARKDOWN = {
    "imports": "## Setup",
    "config": (
        "## Configuration\n"
        "Model, LoRA, and training hyperparameters. `fp16` (not `bf16`) matters here -- a Kaggle "
        "T4 is a Turing GPU (compute capability 7.5); bf16 tensor cores only exist from Ampere "
        "(cc 8.0) onward, so bf16 either silently falls back to slow emulated math or mismatches "
        "the Trainer's mixed-precision mode. See the project README for full sizing rationale."
    ),
    "load & quantize base model": (
        "## Load & quantize the base model\n"
        "Downloads `Qwen/Qwen3-8B` and loads it straight into 4-bit NF4 (double-quantized) via "
        "`bitsandbytes` -- this is the step that needs a real GPU. Expect this cell to take a few "
        "minutes (model download + quantization)."
    ),
    "configure LoRA": (
        "## Attach LoRA adapters\n"
        "`prepare_model_for_kbit_training` is required glue when combining 4-bit loading + "
        "gradient checkpointing (it enables `requires_grad` on the input embeddings and casts "
        "norms to fp32) -- skipping it is a well-known QLoRA gotcha. LoRA targets all 7 linear "
        "projections (attention + MLP), which the QLoRA paper found necessary to match "
        "full-finetune quality; at r=16 across 36 layers this is only ~0.5% of the 8B params."
    ),
    "load & format data": (
        "## Load & format the Roman-Urdu instruction data\n"
        "`build_splits` pools `Redgerd/roman-urdu-alpaca-qa-mix`, deduplicates it, and isolates "
        "the genuinely-Roman-Urdu rows (see the printed diagnostics below -- the raw dataset has "
        "a real duplication bug, documented in the README, that would otherwise double-count "
        "~488 examples). Each example is formatted through Qwen3's chat template with the prompt "
        "tokens masked out of the loss (`-100`), so the model only learns to predict the response."
    ),
    "train": "## Train\nWatch the loss go down here -- this is the actual fine-tuning step.",
    "plot training loss": "## Training loss curve\nA picture of the cell above -- train/eval loss vs. step.",
    "save adapter": (
        "## Save the adapter\n"
        "Only the LoRA adapter is saved (tens of MB), not a merged model -- keep this directory "
        "(or push it to the HF Hub) to reuse the fine-tuned assistant later via `inference.py` / `app/app.py`."
    ),
    "evaluate: base vs tuned + ROUGE-L": (
        "## Evaluate: base (zero-shot) vs QLoRA-tuned\n"
        "Both arms are generated from the *same* loaded model, toggling `peft`'s "
        "`disable_adapter()` context manager -- avoids loading two full copies of an 8B model on "
        "one T4. Scored with ROUGE-L (`use_stemmer=False` -- the default English stemmer would "
        "mangle Roman Urdu tokens)."
    ),
    "plot rouge comparison": "## ROUGE-L: base vs tuned, at a glance\nA picture of the cell above's two score dicts.",
    "sample completions": (
        "## See it for yourself\n"
        "A handful of held-out prompts, generated by both models side by side -- this is the "
        "part worth actually reading, since a single ROUGE-L number undersells what ~485 "
        "training examples can and can't teach a model."
    ),
}


_cell_counter = 0


def _next_id() -> str:
    global _cell_counter
    _cell_counter += 1
    return f"cell-{_cell_counter}"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": source.splitlines(keepends=True)}


def main():
    qlora_utils_code = strip_module_docstring(read("qlora_utils.py"))
    metrics_utils_code = strip_module_docstring(read("metrics_utils.py"))
    data_prep_code = strip_default_out_dir(strip_local_imports(strip_main_block(strip_module_docstring(read("data_prep.py")))))
    train_kaggle_code = strip_local_imports(strip_module_docstring(read("train_kaggle.py")))

    cells = [
        md_cell(
            "# Roman Urdu QLoRA Assistant — Training (Kaggle GPU)\n\n"
            "Auto-generated from `src/` by `scripts/build_kaggle_notebook.py` — "
            "**do not hand-edit this notebook**; change the source files and rerun the "
            "builder instead.\n\n"
            "Before running: Settings → Accelerator → GPU T4 x1, and Internet → On. "
            "Run cells top to bottom and read the markdown between them -- this notebook is "
            "meant to be watched, not just executed blind."
        ),
        code_cell(
            '!pip install -q -U "transformers>=4.51" "peft>=0.11" "bitsandbytes>=0.43" '
            '"datasets>=2.19" accelerate rouge-score matplotlib huggingface_hub\n'
        ),
        code_cell(qlora_utils_code),
        code_cell(metrics_utils_code),
        code_cell(data_prep_code),
    ]

    for name, block in split_by_section_markers(train_kaggle_code):
        if name in SECTION_MARKDOWN:
            cells.append(md_cell(SECTION_MARKDOWN[name]))
        cells.append(code_cell(block))

    cells.append(
        md_cell(
            "Outputs are written under `/kaggle/working/outputs` (the LoRA adapter) and "
            "`/kaggle/working/results` (`metrics.json`, `sample_completions.json`). Pull them back "
            "locally with:\n\n"
            "```\nkaggle kernels output <username>/<slug> -p ./kaggle_output\n```\n\n"
            "Or push the adapter directly to the Hugging Face Hub from within this kernel with "
            "`model.push_to_hub(...)` / `tokenizer.push_to_hub(...)` if you'd rather skip the pull step."
        )
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
