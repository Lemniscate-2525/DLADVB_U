from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
VAL_FILE = PROCESSED_DIR / "val.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

MODEL_NAME = "distilbert-base-uncased"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 42

MAX_SEQ_LEN = 128

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

EPOCHS = 4

NUM_LABELS = 2

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

TARGET_MODULES = [
    "q_lin",
    "v_lin"
]

OUTPUT_DIR = CHECKPOINT_DIR / "model"

TOKENIZER_DIR = CHECKPOINT_DIR / "tokenizer"

BEST_MODEL_NAME = "best_model"

NUM_WORKERS = 4

PIN_MEMORY = True

LOG_INTERVAL = 100

SAVE_BEST_ONLY = True