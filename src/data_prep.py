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


