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

