"""Lightweight evaluation helpers for plain-language rewrites."""

from __future__ import annotations

import re
from dataclasses import dataclass

from medexplain.clinical_examples import extract_jargon


@dataclass(frozen=True)
class RewriteMetrics:
    """Simple metrics that support before/after comparison."""

    word_count: int
    sentence_count: int
    average_sentence_length: float
    jargon_count: int
    flesch_reading_ease: float


def tokenize_words(text: str) -> list[str]:
    """Split text into lowercase word tokens."""

    return re.findall(r"[a-zA-Z]+", text.lower())


def count_sentences(text: str) -> int:
    """Estimate sentence count."""

    sentences = [part for part in re.split(r"[.!?]+", text.strip()) if part.strip()]
    return max(1, len(sentences))


def estimate_syllables(word: str) -> int:
    """Estimate syllables with a small heuristic suitable for comparison."""

    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    """Compute an approximate Flesch reading ease score."""

    words = tokenize_words(text)
    if not words:
        return 0.0
    sentence_count = count_sentences(text)
    syllables = sum(estimate_syllables(word) for word in words)
    score = 206.835 - 1.015 * (len(words) / sentence_count) - 84.6 * (syllables / len(words))
    return round(score, 1)


def count_jargon(text: str) -> int:
    """Count known glossary concepts in text."""

    return len(extract_jargon(text))


def evaluate_rewrite(text: str) -> RewriteMetrics:
    """Compute simple readability and jargon metrics for a rewrite."""

    words = tokenize_words(text)
    sentence_count = count_sentences(text)
    word_count = len(words)
    avg_sentence_length = round(word_count / sentence_count, 1) if sentence_count else 0.0
    return RewriteMetrics(
        word_count=word_count,
        sentence_count=sentence_count,
        average_sentence_length=avg_sentence_length,
        jargon_count=count_jargon(text),
        flesch_reading_ease=flesch_reading_ease(text),
    )
