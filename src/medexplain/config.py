"""Shared project paths and model settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "medical_rewrites_seed.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "medical_rewrites_train.csv"
OUTPUT_DIR = DATA_DIR / "outputs"
DEFAULT_MODEL_NAME = "google/flan-t5-small"
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "models" / "medexplain-lora"


@dataclass(frozen=True)
class GenerationConfig:
    """Settings used for rewrite generation."""

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9
    repetition_penalty: float = 1.1


def adapter_dir_from_env() -> Path:
    """Return the adapter directory configured for the current environment."""

    configured = os.getenv("MEDEXPLAIN_MODEL_DIR")
    if configured:
        return Path(configured)
    return DEFAULT_ADAPTER_DIR
