"""Fine-tune the MedExplain rewrite model with LoRA.

AI assistance was used to draft this educational scaffold. Core libraries:
https://huggingface.co/docs/transformers and https://huggingface.co/docs/peft
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def tokenize_batch(batch: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenize model inputs and labels."""

    prompts = [build_rewrite_prompt(text) for text in batch["source_text"]]
    model_inputs = tokenizer(prompts, max_length=512, truncation=True)
    labels = tokenizer(text_target=batch["target_text"], max_length=160, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_datasets(data: pd.DataFrame, tokenizer: AutoTokenizer) -> tuple[Dataset, Dataset]:
    """Split and tokenize the rewrite examples."""

    train_df, eval_df = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))
    train_dataset = train_dataset.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    eval_dataset = eval_dataset.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    return train_dataset, eval_dataset


def main() -> None:
    """Train and save a LoRA adapter."""

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
    train_dataset, eval_dataset = build_datasets(data, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="data/outputs/training",
        learning_rate=5e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=80,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
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
        "training_examples": len(data),
        "target_modules": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    }
    (DEFAULT_ADAPTER_DIR / "training_config.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved LoRA adapter to {DEFAULT_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
