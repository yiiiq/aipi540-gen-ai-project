"""Tests for readability and jargon metrics."""

from medexplain.evaluation import count_jargon, evaluate_rewrite, flesch_reading_ease
from medexplain.prompts import build_rewrite_prompt


def test_jargon_count_detects_known_terms() -> None:
    """Known medical jargon should be counted."""

    assert count_jargon("Assessment consistent with STEMI. Cardiology consulted for emergent PCI.") == 2


def test_jargon_count_detects_medical_phrases() -> None:
    """Multi-word clinical phrases should be counted."""

    text = "The patient has pitting edema and bibasilar crackles."
    assert count_jargon(text) == 2


def test_evaluate_rewrite_returns_metrics() -> None:
    """Metric object should contain useful comparison values."""

    metrics = evaluate_rewrite("You have high blood pressure. Ask your care team about your plan.")
    assert metrics.word_count > 0
    assert metrics.sentence_count == 2
    assert metrics.jargon_count == 0


def test_flesch_score_handles_empty_text() -> None:
    """Empty text should not fail metric generation."""

    assert flesch_reading_ease("") == 0.0


def test_rewrite_prompt_uses_instruction_format() -> None:
    """Training and inference should use the same explicit instruction format."""

    prompt = build_rewrite_prompt("Troponin I markedly elevated.")
    assert prompt == (
        "Simplify the following medical text into plain, patient-friendly language at "
        "approximately a 6th-8th grade reading level. Preserve all medical facts, "
        "numbers, and meaning. Input: Troponin I markedly elevated.\nOutput:"
    )
