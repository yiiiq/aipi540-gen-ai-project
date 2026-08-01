"""Prompt templates for medical rewrites."""

from __future__ import annotations


REWRITE_INSTRUCTION = (
    "Simplify the following medical text into plain, patient-friendly language at "
    "approximately a 6th-8th grade reading level. Preserve all medical facts, "
    "numbers, and meaning."
)


def build_rewrite_prompt(text: str) -> str:
    """Create the instruction prompt used for training and inference."""

    return f"{REWRITE_INSTRUCTION} Input: {text.strip()}\nOutput:"
