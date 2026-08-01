"""Fine-tune the MedExplain rewrite model with LoRA.

AI assistance was used to draft this educational scaffold. Core libraries:
https://huggingface.co/docs/transformers and https://huggingface.co/docs/peft
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse

import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medexplain.config import DEFAULT_ADAPTER_DIR, DEFAULT_MODEL_NAME, PROCESSED_DATA_PATH
from medexplain.data import load_rewrite_data
from medexplain.prompts import build_rewrite_prompt


def parse_args() -> argparse.Namespace:
    """Parse training options."""

    parser = argparse.ArgumentParser(description="Train the MedExplain LoRA adapter.")
    parser.add_argument("--epochs", type=float, default=8, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps. Defaults to an effective batch size of 8.",
    )
    parser.add_argument("--max-source-length", type=int, default=256, help="Maximum tokenized input length.")
    parser.add_argument("--max-target-length", type=int, default=256, help="Maximum tokenized label length.")
    return parser.parse_args()


def tokenize_batch(
    batch: dict,
    tokenizer: AutoTokenizer,
    max_source_length: int,
    max_target_length: int,
) -> dict:
    """Tokenize model inputs and labels."""

    prompts = [build_rewrite_prompt(text) for text in batch["source_text"]]
    model_inputs = tokenizer(prompts, max_length=max_source_length, truncation=True)
    labels = tokenizer(text_target=batch["target_text"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_datasets(
    data: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_source_length: int,
    max_target_length: int,
) -> tuple[Dataset, Dataset]:
    """Split and tokenize the rewrite examples."""

    train_df, eval_df = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))
    tokenize = lambda batch: tokenize_batch(batch, tokenizer, max_source_length, max_target_length)
    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    return train_dataset, eval_dataset


def main() -> None:
    """Train and save a LoRA adapter."""

    args = parse_args()
    data = load_rewrite_data(PROCESSED_DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL_NAME)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    )
    model = get_peft_model(base_model, lora_config)
    train_dataset, eval_dataset = build_datasets(
        data,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="data/outputs/training",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        predict_with_generate=True,
        fp16=False,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    DEFAULT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DEFAULT_ADAPTER_DIR)
    tokenizer.save_pretrained(DEFAULT_ADAPTER_DIR)
    metadata = {
        "base_model": DEFAULT_MODEL_NAME,
        "adaptation": "LoRA",
        "instruction_format": (
            "Simplify the following medical text into plain, patient-friendly language at "
            "approximately a 6th-8th grade reading level. Preserve all medical facts, "
            "numbers, and meaning. Input: {expert}\nOutput:"
        ),
        "training_examples": len(data),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "target_modules": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    }
    (DEFAULT_ADAPTER_DIR / "training_config.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved LoRA adapter to {DEFAULT_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
