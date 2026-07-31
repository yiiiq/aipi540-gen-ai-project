"""Create the processed training dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medexplain.config import PROCESSED_DATA_PATH, RAW_DATA_PATH
from medexplain.data import save_processed_data


def main() -> None:
    """Validate and save the processed dataset."""

    data = save_processed_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print(f"Saved {len(data)} examples to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
