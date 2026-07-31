"""Dataset loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
        RewriteExample(
            source_text="A disrupted ossicular chain may be repaired during tympanoplasty as well.",
            target_text="If the ossicles have been damaged, they may be repaired at the same time.",
        ),
        RewriteExample(
            source_text="Abdominal pain (usually starting in one quadrant and spreading to the whole abdomen) occurs in about 95 % of patients and can vary in severity with each attack.",
            target_text="The pain usually starts in one part of the abdomen, then spreads throughout the entire abdomen. The severity of the pain may vary with each attack.",
        ),
        RewriteExample(
            source_text="About 50 % of patients report constitutional symptoms such as fever, malaise, night sweats, weight loss, fatigue, and/or arthralgias.",
            target_text="Sometimes the disorder begins with fever, muscle and joint aches, loss of appetite, weight loss, and night sweats.",
        ),
    ]
