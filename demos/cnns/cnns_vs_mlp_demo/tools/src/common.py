"""Shared paths, constants and helpers for the shifted-MNIST experiment."""

import json
import os
import platform
import random
from pathlib import Path

# Needed by torch.use_deterministic_algorithms() for cuBLAS; must be set before CUDA init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"  # torchvision MNIST lives in data/MNIST
OUTPUT_DIR = EXP_DIR / "outputs"  # checkpoints, metrics, eval results (local only)
CKPT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
EVAL_DIR = OUTPUT_DIR / "eval"
EXPORT_DIR = EXP_DIR / "export"  # deliverables for the browser demo

CANVAS = 56
DIGIT = 28
MAX_OFFSET = CANVAS - DIGIT  # 28 -> offsets 0..28, 29 positions per axis
N_OFFSETS = MAX_OFFSET + 1

ARCHS = ("mlp", "cnn_fc", "cnn_gap")
REGIMES = ("tl", "all")
SEEDS = (0, 1, 2)
EXPORT_SEED = 0
# Seed for the train/val split, val offsets, eval subset and test offsets (all models).
SPLIT_SEED = 0
N_VAL = 5000
SUBSET_PER_CLASS = 200


def model_id(arch: str, regime: str) -> str:
    return f"{arch}_{regime}"


def run_id(arch: str, regime: str, seed: int) -> str:
    return f"{arch}_{regime}_s{seed}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Full fp32 everywhere, so float32 vs. float16-weight comparisons are meaningful.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{device.type} ({platform.processor() or platform.machine()})"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())
