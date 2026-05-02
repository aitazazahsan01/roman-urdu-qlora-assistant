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
