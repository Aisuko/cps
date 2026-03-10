from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS_DIR = RESULTS_DIR / "paper_splits"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

DEFAULT_DATASET = "NQ-Swap"
DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"

CHG_NUM_MASKS = 10
CHG_NUM_UPDATES = 50
CHG_BATCH_SIZE = 32
CHG_THRESHOLD = 0.5
CHG_FIT_SAMPLE_SIZE = 40

MAX_INPUT_TOKENS = 1536
MAX_NEW_TOKENS = 20

TOPK_GRID = (2, 4, 6, 8, 10, 12, 16)
MASK_SCALE = 0.0
ABLATION_DELTA = 0.3
DELTA_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)

SPLIT_CHG_FIT_ROWS = 40
SPLIT_CONTROLLER_TRAIN_ROWS = 20
SPLIT_SELECTION_DEV_ROWS = 100
SPLIT_TEST_ROWS = 100


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_env() -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    except ImportError:
        pass
    token = os.getenv("hf_token") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
    return token
