    disagreements = []
    for i, row in enumerate(pool):
        in_membership = _row_key(row) in membership_keys
        heuristic = _looks_like_roman_urdu(row["instruction"], row["output"])
        if in_membership != heuristic:
            disagreements.append((row["instruction"][:80], in_membership, heuristic))
        if in_membership and heuristic:
            keep_idx.append(i)

    if disagreements:
        print(f"  {len(disagreements)} membership/heuristic disagreement(s), dropped:")
        for instr, mem, heur in disagreements:
            print(f"    membership={mem} heuristic={heur}  {instr!r}")

    return pool.select(keep_idx)


MAX_ROW_KEY_FIELDS = ("instruction", "input", "output")


def _row_key(row: dict) -> tuple:
    return tuple(row.get(f, "") for f in MAX_ROW_KEY_FIELDS)


def load_raw_pool() -> Dataset:
    return load_dataset(DATASET_NAME)["train"]


def _dedupe(pool: Dataset) -> Dataset:
    seen = set()
    keep_idx = []
    for i, row in enumerate(pool):
        key = _row_key(row)
        if key not in seen:
        print(
            f"  chat-template token lengths: min={lengths[0]} p50={lengths[n // 2]} "
            f"p90={lengths[int(n * 0.9)]} p99={lengths[int(n * 0.99)]} max={lengths[-1]}"
        )
        if lengths[-1] > MAX_SEQ_LENGTH:
            print(f"  warning: max length {lengths[-1]} exceeds MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} -- those rows will be truncated")
    except Exception as exc:
        print(f"  warning: couldn't load {BASE_MODEL_NAME} tokenizer to measure lengths ({exc}); skipping this check")

    print("Sanity check passed.")


def build_splits(val_fraction: float, seed: int) -> tuple:
    print(f"Loading {DATASET_NAME} from the Hugging Face Hub...")
    raw = load_raw_pool()
    print(f"  raw train split: {len(raw)} rows")

    with open(args.out_dir / "filter_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved processed splits + filter_report.json to {args.out_dir}")


if __name__ == "__main__":
    main()
    print(f"  {RU_SOURCE_FILE}: {n_lines} lines, {len(keys)} unique keys ({n_lines - len(keys)} internal duplicate(s))")
    return keys


def _looks_like_roman_urdu(instruction: str, output: str) -> bool:
    text = f"{instruction} {output}".lower()
    words = text.split()
    if not words:
        return False
    hits = sum(1 for w in words if w.strip(".,?!\"'") in RU_STOPWORDS)
    return (hits / len(words)) > 0.03


def filter_roman_urdu(pool: Dataset) -> Dataset:
    membership_keys = _roman_urdu_membership_keys()

    keep_idx = []
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    train, val = build_splits(args.val_fraction, args.seed)
    print(f"train={len(train)}  validation={len(val)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(args.out_dir / "train"))
    val.save_to_disk(str(args.out_dir / "validation"))

    report = {
        "raw_rows": len(load_raw_pool()),
        "final_train": len(train),
        "final_validation": len(val),
    }
            seen.add(key)
            keep_idx.append(i)
    return pool.select(keep_idx)


def _roman_urdu_membership_keys() -> set:
    path = hf_hub_download(repo_id=DATASET_NAME, filename=RU_SOURCE_FILE, repo_type="dataset")
    keys = set()
    n_lines = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            row = json.loads(line)
            keys.add(_row_key(row))
"""Build a clean Roman-Urdu-only instruction-tuning set from
Redgerd/roman-urdu-alpaca-qa-mix.

The Hub's `train` split (1,489 rows) is actually two raw JSONL files
concatenated: `combined_roman_urdu_english.jsonl` (1000 rows, ~500 Roman Urdu +
~500 English Alpaca, shuffled) followed by `roman_urdu_QA_full_alpaca.jsonl`
(489 rows, the pure-Roman-Urdu source the combined file's RU half was drawn
from) -- verified by downloading both raw files and diffing them against the
Hub's auto-converted parquet byte-for-byte. 488 of those 489 "full" rows are
exact-content duplicates of rows already inside `combined`, so the naive
1489-row split silently double-counts ~488 examples. There is also no
language-tag column (parquet columns are only instruction/input/output/text),
so isolating the Roman-Urdu-only rows needs two independent signals:

  1. File-membership: does this row's (instruction, input, output) match a row
     in the raw roman_urdu_QA_full_alpaca.jsonl source file?
  2. A lightweight Roman-Urdu stopword-hit-rate heuristic, as a cross-check.
def sanity_check(ds: Dataset) -> None:
    n_empty = sum(1 for o in ds["output"] if not o.strip())
    if n_empty:
        raise ValueError(f"{n_empty} rows have an empty output")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        lengths = []
        for row in ds:
            messages = [{"role": "user", "content": row["instruction"] + (("\n\n" + row["input"]) if row["input"].strip() else "")}]
            messages.append({"role": "assistant", "content": row["output"]})
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)["input_ids"]
            lengths.append(len(ids))
        lengths.sort()
        n = len(lengths)

Cross-validating the two found only 3 disagreements out of 999 deduped rows,
all the same failure mode: English-language *questions about* Urdu (e.g. "Give
some examples of Singular and Plural in Urdu Language") that happen to live in
the "pure Roman-Urdu" source file. Those 3 are dropped.

Usage:
    python src/data_prep.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
    deduped = _dedupe(raw)
    print(f"  after dedup by (instruction, input, output): {len(deduped)} rows")

    ru_only = filter_roman_urdu(deduped)
    print(f"  after Roman-Urdu file-membership + stopword-heuristic filter: {len(ru_only)} rows")

    sanity_check(ru_only)

    shuffled = ru_only.shuffle(seed=seed)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val = shuffled.select(range(n_val))
    train = shuffled.select(range(n_val, len(shuffled)))
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
