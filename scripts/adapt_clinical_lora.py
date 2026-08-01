"""Further adapt the LoRA adapter on curated clinical-note demo examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
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

from medexplain.clinical_examples import CLINICAL_EXAMPLES
from medexplain.config import DEFAULT_ADAPTER_DIR, DEFAULT_MODEL_NAME
from medexplain.prompts import build_rewrite_prompt


def parse_args() -> argparse.Namespace:
    """Parse clinical adaptation options."""

    parser = argparse.ArgumentParser(description="Adapt the LoRA adapter on clinical-note examples.")
    parser.add_argument("--epochs", type=float, default=10, help="Number of clinical adaptation epochs.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=40, help="How often to repeat the 10 curated examples.")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    return parser.parse_args()


def tokenize_batch(
    batch: dict,
    tokenizer: AutoTokenizer,
    max_source_length: int,
    max_target_length: int,
) -> dict:
    """Tokenize clinical-note prompts and plain-English targets."""

    prompts = [build_rewrite_prompt(text) for text in batch["source_text"]]
    model_inputs = tokenizer(prompts, max_length=max_source_length, truncation=True)
    labels = tokenizer(text_target=batch["target_text"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer) -> Dataset:
    """Create a repeated supervised dataset from curated clinical examples."""

    rows = [
        {
            "source_text": example.source_text,
            "target_text": example.plain_english,
        }
        for _ in range(args.repeat)
        for example in CLINICAL_EXAMPLES
    ]
    dataset = Dataset.from_list(rows)
    return dataset.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
            args.max_source_length,
            args.max_target_length,
        ),
        batched=True,
    )


def load_or_create_adapter() -> tuple[object, object, bool]:
    """Load the existing adapter when present, otherwise initialize a new one."""

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL_NAME)
    adapter_active = DEFAULT_ADAPTER_DIR.exists() and any(DEFAULT_ADAPTER_DIR.iterdir())
    if adapter_active:
        model = PeftModel.from_pretrained(base_model, str(DEFAULT_ADAPTER_DIR), is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        )
        model = get_peft_model(base_model, lora_config)
    return model, tokenizer, adapter_active


def update_metadata(args: argparse.Namespace, adapter_was_active: bool) -> None:
    """Record the clinical adaptation stage in adapter metadata."""

    metadata_path = DEFAULT_ADAPTER_DIR / "training_config.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    metadata["clinical_adaptation"] = {
        "source": "src/medexplain/clinical_examples.py",
        "examples": len(CLINICAL_EXAMPLES),
        "repeat": args.repeat,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "adapter_was_active_before_stage": adapter_was_active,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    """Run clinical-note adaptation and save the adapter."""

    args = parse_args()
    model, tokenizer, adapter_was_active = load_or_create_adapter()
    train_dataset = build_dataset(args, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="data/outputs/clinical-adaptation",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=10,
        predict_with_generate=True,
        fp16=False,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    DEFAULT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DEFAULT_ADAPTER_DIR)
    tokenizer.save_pretrained(DEFAULT_ADAPTER_DIR)
    update_metadata(args, adapter_was_active)
    print(f"Saved clinically adapted LoRA adapter to {DEFAULT_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
