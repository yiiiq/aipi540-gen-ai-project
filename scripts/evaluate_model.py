"""Generate before/after examples for the project pitch."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medexplain.config import DEFAULT_ADAPTER_DIR, DEFAULT_MODEL_NAME, OUTPUT_DIR, PROCESSED_DATA_PATH
from medexplain.data import load_rewrite_data
from medexplain.evaluation import evaluate_rewrite
from medexplain.model import generate_rewrite, load_model_with_optional_adapter


def metric_columns(prefix: str, text: str) -> dict[str, float | int]:
    """Return flattened metric values with a prefix."""

    metrics = evaluate_rewrite(text)
    return {
        f"{prefix}_word_count": metrics.word_count,
        f"{prefix}_avg_sentence_length": metrics.average_sentence_length,
        f"{prefix}_jargon_count": metrics.jargon_count,
        f"{prefix}_flesch": metrics.flesch_reading_ease,
    }


def main() -> None:
    """Create a CSV with base and fine-tuned model outputs."""

    data = load_rewrite_data(PROCESSED_DATA_PATH).head(6)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL_NAME)
    tuned_model, tuned_tokenizer, adapter_active = load_model_with_optional_adapter(DEFAULT_ADAPTER_DIR)
    rows = []
    for _, row in data.iterrows():
        source_text = row["source_text"]
        base_output = generate_rewrite(source_text, base_model, tokenizer)
        tuned_output = generate_rewrite(source_text, tuned_model, tuned_tokenizer)
        rows.append(
            {
                "source_text": source_text,
                "reference_rewrite": row["target_text"],
                "base_model_output": base_output,
                "fine_tuned_output": tuned_output,
                "adapter_active": adapter_active,
                **metric_columns("source", source_text),
                **metric_columns("base", base_output),
                **metric_columns("fine_tuned", tuned_output),
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "before_after_examples.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved before/after examples to {output_path}")


if __name__ == "__main__":
    main()
