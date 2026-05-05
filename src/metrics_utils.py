    if _is_notebook():
        plt.show()
    plt.close()
    print(f"Saved ROUGE-L comparison chart to {output_path}")
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar([i - width / 2 for i in x], base_vals, width, label="Base (zero-shot)")
    plt.bar([i + width / 2 for i in x], tuned_vals, width, label="QLoRA-tuned")
    plt.xticks(list(x), labels)
    plt.ylabel("ROUGE-L")
    plt.title("Base vs QLoRA-tuned: ROUGE-L on held-out prompts")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
