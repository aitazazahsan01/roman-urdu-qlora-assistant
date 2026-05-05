```

**5. Run the demo (needs CUDA):**

```bash
python app/app.py --adapter-dir code-aitazaz/roman-urdu-qlora-qwen3-8b
```

Both accept a local path or a Hub repo id interchangeably (`peft` resolves
either transparently) — the trained adapter is public on the Hub, so neither
command actually needs the local `kaggle_output/` folder or your own Kaggle
run to work, as long as you have a CUDA machine to run them on.

## Results

Real numbers from an actual Kaggle T4 run (3 epochs, 438 train / 48 val rows,
ROUGE-L on 40 held-out prompts — see `results/metrics.json` for the raw
numbers and `results/sample_completions.json` for all 8 saved transcripts).

| Model | ROUGE-L Precision | ROUGE-L Recall | ROUGE-L F-measure |
|---|:-:|:-:|:-:|
| Base (zero-shot) | 0.194 | 0.128 | 0.095 |
| **QLoRA-tuned** | **0.211** | **0.174** | **0.164** |

<details open>
<summary><b>Model, quantization, LoRA, training, and evaluation design — click to collapse</b></summary>

<br>

- **Base model**: [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) —
  Apache-2.0, already instruction/chat-tuned, dense (not MoE) transformer
  architecture with a multilingual tokenizer. Its chat template supports an
  `enable_thinking` toggle for extended reasoning traces; this project uses
  `enable_thinking=False` throughout (training *and* inference) for
  straightforward single-turn instruction-following, not chain-of-thought.
- **Method**: 4-bit QLoRA. `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.float16)` — **float16, not bfloat16**: a Kaggle
  T4 is a Turing GPU (compute capability 7.5), and bf16 tensor-core support
  only starts at Ampere (cc 8.0). Most QLoRA recipes default to bf16 because
  they assume A100-class hardware; on a T4 that either silently falls back to
  slow emulated math or mismatches the Trainer's fp16 mixed-precision mode.
  Paired with `TrainingArguments(fp16=True)`.
- **LoRA config**: rank 16, alpha 32, dropout 0.05, targeting all 7 linear
  projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
  down_proj`) — not just attention. The QLoRA paper found this is needed to
  match full-finetune quality, and it's essentially free here: ~44M trainable
  params, about 0.5% of the 8B total. (Qwen3's `q_norm`/`k_norm` QK-norm
  formatting, loss masking, and LoRA adapter attachment against a tiny
  random-weight Qwen3-architecture model in full precision on CPU — it does
  **not**, and cannot, validate the real 4-bit quantized path. That can only
  be exercised on a real CUDA machine, which is exactly what
  `kaggle/train_kernel.ipynb` is for.

</details>

## Model on the Hugging Face Hub

The trained adapter is public and loadable directly (no local training or
Kaggle run required):
[`code-aitazaz/roman-urdu-qlora-qwen3-8b`](https://huggingface.co/code-aitazaz/roman-urdu-qlora-qwen3-8b) —
its model card includes a full usage snippet and the same honest
results/limitations writeup as below.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", quantization_config=..., device_map="auto")
model = PeftModel.from_pretrained(base, "code-aitazaz/roman-urdu-qlora-qwen3-8b")
```

python scripts/build_kaggle_notebook.py   # regenerate kaggle/train_kernel.ipynb if you changed src/
```

Then either:
- Open `kaggle/train_kernel.ipynb` directly on kaggle.com (upload it, or
  create a new notebook and paste it in), set **Settings → Accelerator → GPU
  T4 x1** and **Internet → On**, and **Run All** — read the markdown between
  cells as it goes; the notebook is written to be watched, not just executed
  blind. The last cell prints several instruction / reference / base-model /
  tuned-model transcripts side by side so you can see the adapter's effect
  immediately.
- Or push it via the Kaggle CLI, same mechanism the sibling projects use
  (entirely optional, just a faster way to get the notebook onto Kaggle):
  ```bash
  cd kaggle
  kaggle kernels push --accelerator NvidiaTeslaT4
  kaggle kernels output <username>/roman-urdu-qlora-assistant-training -p ../kaggle_output
  ```

**4. Evaluate a trained adapter (needs CUDA):**

```bash
python src/evaluate.py --adapter-dir code-aitazaz/roman-urdu-qlora-qwen3-8b
# or a local path: --adapter-dir kaggle_output/outputs/qwen3-8b-roman-urdu-qlora
1. **File membership** — does this row's `(instruction, input, output)` match
   a row in the raw `roman_urdu_QA_full_alpaca.jsonl` source file?
2. **A lightweight Roman-Urdu stopword-hit-rate heuristic**, as a cross-check.

Cross-validating the two found only 3 disagreements out of 999 deduped rows —
all the same failure mode: English-language *questions about* Urdu (e.g.
"Give some examples of Singular and Plural in Urdu Language") that happen to
live inside the "pure Roman-Urdu" source file. Those 3 are dropped.

**Net usable pool: ~485 Roman-Urdu rows** (raw 1,489 → 999 deduped → 488
file-matched → 485 after the heuristic cross-check). `data_prep.py` prints
this whole funnel when you run it. Also worth noting: the 489-row source file
itself contains exactly one internal duplicate.

A larger Urdu-*script* (not Roman) Alpaca-52K translation exists —
[`saillab/alpaca-urdu-cleaned`](https://huggingface.co/datasets/saillab/alpaca-urdu-cleaned),
CC-BY-NC academic-only — but transliterating it to Roman Urdu at scale is real
engineering effort with real quality risk, and is **not implemented here**;
it's a documented future-augmentation idea, not a v1 feature.

</details>

## Approach

