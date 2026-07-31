"""Prompt templates for medical rewrites."""

from __future__ import annotations


SYSTEM_STYLE_RULES = (
    "Rewrite the medical text for a patient. Use plain language, keep the meaning, "
    "avoid diagnosis or treatment advice, and mention when the patient should ask "
    "their care team for clarification."
)


def build_rewrite_prompt(text: str) -> str:
    """Create the instruction prompt used for training and inference."""

    return f"{SYSTEM_STYLE_RULES}\n\nMedical text: {text.strip()}\n\nPatient-friendly rewrite:"
