"""Create the processed training dataset.

AI assistance: this file was created with OpenAI Codex. See AI_USAGE.md.
Dataset source option: https://huggingface.co/datasets/cbasu/Med-EASi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medexplain.config import (
    HF_CACHE_DIR,
    MEDEASI_DATASET_NAME,
    MEDEASI_RAW_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from medexplain.data import load_rewrite_data, save_processed_data


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="Build rewrite training data.")
    parser.add_argument(
        "--source",
        choices=["med-easi", "seed"],
        default="med-easi",
        help="Dataset source to process. Defaults to Med-EASi.",
    )
    parser.add_argument("--split", default="train", help="Hugging Face split for Med-EASi.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Maximum Med-EASi rows to keep for fast hackathon training. Use 0 for all rows.",
    )
    return parser.parse_args()


def build_from_med_easi(split: str, max_rows: int) -> pd.DataFrame:
    """Download Med-EASi and convert Expert/Simple columns to project format."""

    from datasets import load_dataset

    dataset = load_dataset(MEDEASI_DATASET_NAME, split=split, cache_dir=str(HF_CACHE_DIR))
    data = dataset.to_pandas()[["Expert", "Simple"]].rename(
        columns={"Expert": "source_text", "Simple": "target_text"}
    )
    data = data.dropna().drop_duplicates().reset_index(drop=True)
    if max_rows > 0:
        data = data.head(max_rows)
    MEDEASI_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(MEDEASI_RAW_PATH, index=False)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH, index=False)
    return load_rewrite_data(PROCESSED_DATA_PATH)


def main() -> None:
    """Validate and save the processed dataset."""

    args = parse_args()
    if args.source == "med-easi":
        data = build_from_med_easi(args.split, args.max_rows)
    else:
        data = save_processed_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print(f"Saved {len(data)} examples to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
