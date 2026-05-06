

def run(*args):
import sys
from pathlib import Path

"""One-shot local pipeline sanity check: data prep -> tiny CPU QLoRA training
run -> evaluation -> inference, to catch bugs before ever spending Kaggle GPU
quota.


def main():

Usage:
    python scripts/smoke_test.py
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SMOKE_OUTPUT_DIR = ROOT / "outputs" / "smoke-test-qlora"
        "yourself on a Kaggle T4."
    )

