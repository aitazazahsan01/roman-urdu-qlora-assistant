

def run(*args):
import sys
from pathlib import Path

"""One-shot local pipeline sanity check: data prep -> tiny CPU QLoRA training
run -> evaluation -> inference, to catch bugs before ever spending Kaggle GPU
quota.


def main():
