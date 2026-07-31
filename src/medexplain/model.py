"""Model loading and generation helpers.

AI assistance was used to draft this educational scaffold. Core libraries:
https://huggingface.co/docs/transformers and https://huggingface.co/docs/peft
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from medexplain.config import DEFAULT_MODEL_NAME, GenerationConfig
from medexplain.prompts import build_rewrite_prompt


def load_tokenizer(model_name_or_path: str = DEFAULT_MODEL_NAME) -> Any:
    """Load the tokenizer used by the base model."""

    return AutoTokenizer.from_pretrained(model_name_or_path)


def load_base_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load the base sequence-to-sequence model."""

    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


def load_model_with_optional_adapter(
    adapter_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[Any, Any, bool]:
    """Load the base model and attach a LoRA adapter when available."""

    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name)
    adapter_active = adapter_dir.exists() and any(adapter_dir.iterdir())
    if adapter_active:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer, adapter_active


def generate_rewrite(
    text: str,
    model: Any,
    tokenizer: Any,
    generation_config: GenerationConfig | None = None,
) -> str:
    """Generate a patient-friendly rewrite for the provided text."""

    config = generation_config or GenerationConfig()
    prompt = build_rewrite_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generation_kwargs = asdict(config)
    if generation_kwargs["temperature"] <= 0:
        generation_kwargs["do_sample"] = False
        generation_kwargs.pop("temperature")
        generation_kwargs.pop("top_p")
    else:
        generation_kwargs["do_sample"] = True
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
