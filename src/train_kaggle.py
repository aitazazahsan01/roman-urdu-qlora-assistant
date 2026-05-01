    plot_training_loss,
    prepare_for_training,
)

# ---- config ----
VAL_FRACTION = 0.1
SEED = 42
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
N_SAMPLE_COMPLETIONS = 8

ON_KAGGLE = Path("/kaggle/working").exists()
OUTPUT_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path(__file__).resolve().parent.parent
