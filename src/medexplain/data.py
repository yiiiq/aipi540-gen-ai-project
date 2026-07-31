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
            source_text="Take this medication BID with meals and monitor for adverse gastrointestinal effects.",
            target_text="Take this medicine twice a day with food. Call your care team if you have stomach pain, nausea, vomiting, or diarrhea.",
        ),
        RewriteExample(
            source_text="The imaging study shows no acute cardiopulmonary abnormality.",
            target_text="Your scan does not show a new serious problem with your heart or lungs.",
        ),
        RewriteExample(
            source_text="Use the inhaler PRN for dyspnea or wheezing.",
            target_text="Use the inhaler when needed for shortness of breath or wheezing. Ask your care team how often it is safe to use.",
        ),
    ]
