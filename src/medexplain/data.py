"""Dataset loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from medexplain.clinical_examples import CLINICAL_EXAMPLES

REQUIRED_COLUMNS = ("source_text", "target_text")


@dataclass(frozen=True)
class RewriteExample:
    """One paired medical rewrite example."""

    source_text: str
    target_text: str


def load_rewrite_data(path: Path) -> pd.DataFrame:
    """Load paired rewrite examples from a CSV file."""

    data = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = data.loc[:, REQUIRED_COLUMNS].dropna()
    data["source_text"] = data["source_text"].astype(str).str.strip()
    data["target_text"] = data["target_text"].astype(str).str.strip()
    return data[(data["source_text"] != "") & (data["target_text"] != "")].reset_index(drop=True)


def save_processed_data(raw_path: Path, processed_path: Path) -> pd.DataFrame:
    """Validate the raw CSV and save a clean processed training file."""

    data = load_rewrite_data(raw_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_path, index=False)
    return data


def default_examples() -> list[RewriteExample]:
    """Return examples for the web app sample selector."""

    return [
        RewriteExample(source_text=example.source_text, target_text=example.plain_english)
        for example in CLINICAL_EXAMPLES
    ]
