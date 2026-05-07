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
