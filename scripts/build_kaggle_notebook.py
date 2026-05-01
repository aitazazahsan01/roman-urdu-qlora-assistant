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
