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
