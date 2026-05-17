"""One-shot local pipeline sanity check: data prep -> tiny CPU QLoRA training
run -> evaluation -> inference, to catch bugs before ever spending Kaggle GPU
quota.

Important, honest scope: this validates the data pipeline, chat-template
formatting, loss masking, and LoRA adapter attachment on CPU with a tiny
random-weight model in full precision. It does NOT and cannot validate the
real 4-bit bitsandbytes quantization path -- that requires CUDA and can only
be exercised on the Kaggle GPU kernel itself (kaggle/train_kernel.ipynb).

Usage:
    python scripts/smoke_test.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SMOKE_OUTPUT_DIR = ROOT / "outputs" / "smoke-test-qlora"


def run(*args):
    cmd = [sys.executable, *args]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=SRC, check=True)


def main():
    if not (ROOT / "data" / "processed" / "roman_urdu_qa" / "train" / "dataset_info.json").exists():
        run("data_prep.py")

    run("train_qlora.py", "--smoke-test")
    run("evaluate.py", "--smoke-test")
    print(
        "\nSmoke test passed -- data pipeline, chat-template formatting, loss masking, and "
        "LoRA attachment are all wired up correctly. This did NOT exercise the real 4-bit "
        "quantized path (needs CUDA) -- that only happens when you run kaggle/train_kernel.ipynb "
        "yourself on a Kaggle T4."
    )


if __name__ == "__main__":
    main()
