from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RAW_SMALL_DIR = RAW_DIR / "MIND-small"
RAW_SMALL_TRAIN_DIR = RAW_SMALL_DIR / "train"
RAW_SMALL_DEV_DIR = RAW_SMALL_DIR / "dev"
RAW_SMALL_TEST_DIR = RAW_SMALL_DIR / "test"

DATASET = "small"
SEED = 42
NEG_RATIO = 4
PCA_COMPONENTS = 32
TITLE_EMBED_DIM = 768
TITLE_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
ENTITY_EMBED_DIM = 100
SPLIT_RATIOS = (0.70, 0.15, 0.15)
MAX_BEHAVIOR_ROWS = 5000  # Reduced for MemoryError stopgap; set None for full dataset

# GPU acceleration
GPU_DEVICE = 0                      # CUDA device index
GPU_ENABLED = True                  # Set False to force CPU-only
GPU_BATCH_SIZE = 4096               # Max rows per GPU batch for cosine diversity

_DIRS = [
    DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, NOTEBOOKS_DIR,
    RAW_SMALL_DIR, RAW_SMALL_TRAIN_DIR, RAW_SMALL_DEV_DIR, RAW_SMALL_TEST_DIR,
]
