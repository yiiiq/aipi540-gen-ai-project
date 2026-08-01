"""Streamlit app for MedExplain."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medexplain.config import GenerationConfig, adapter_dir_from_env
from medexplain.clinical_examples import CLINICAL_EXAMPLES, extract_jargon, find_matching_example
from medexplain.evaluation import evaluate_rewrite
from medexplain.model import generate_rewrite, load_model_with_optional_adapter


st.set_page_config(page_title="MedExplain", page_icon="M", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def cached_model(adapter_path: str):
    """Load the model once per Streamlit session."""

    return load_model_with_optional_adapter(Path(adapter_path))


def render_metrics(label: str, text: str) -> None:
    """Render readability and jargon metrics for generated text."""

    metrics = evaluate_rewrite(text)
    st.markdown(f"**{label}**")
    cols = st.columns(4)
    cols[0].metric("Words", metrics.word_count)
    cols[1].metric("Avg sentence", metrics.average_sentence_length)
    cols[2].metric("Jargon terms", metrics.jargon_count)
    cols[3].metric("Flesch score", metrics.flesch_reading_ease)


def render_jargon_table(text: str) -> None:
    """Render known jargon explanations for the current note."""

    matched_example = find_matching_example(text)
    jargon_items = matched_example.jargon if matched_example else extract_jargon(text)
    st.subheader("Jargon Explained")
    if not jargon_items:
        st.info("No known demo jargon terms detected.")
        return
    st.table([{"Jargon": item.term, "Plain English": item.definition} for item in jargon_items])


def main() -> None:
    """Run the Streamlit interface."""

    st.title("MedExplain")
    st.caption("A hybrid LoRA + glossary prototype for patient-friendly clinical-note explanations.")

    adapter_path = adapter_dir_from_env()
    model, tokenizer, adapter_active = cached_model(str(adapter_path))
    if adapter_active:
        st.success(f"Fine-tuned adapter active: {adapter_path}")
    else:
        st.warning(
            "Fine-tuned adapter not found. The app is running the base model only. "
            "Train with `python scripts/train_model.py` before final deployment."
        )

    sample_labels = ["Write my own text"] + [example.title for example in CLINICAL_EXAMPLES]
    selected = st.selectbox("Choose a sample or write your own", sample_labels)
    selected_example = next(
        (example for example in CLINICAL_EXAMPLES if example.title == selected),
        None,
    )
    default_text = "" if selected_example is None else selected_example.source_text
    user_text = st.text_area(
        "Medical text",
        value=default_text,
        height=150,
        placeholder="Paste medical jargon, visit notes, lab language, or discharge instructions here.",
    )

    with st.sidebar:
        st.header("Architecture")
        st.write(
            "The fine-tuned LoRA model generates the plain-English summary. A deterministic glossary layer "
            "explains detected medical terms."
        )
        st.divider()
        st.header("Generation")
        max_new_tokens = st.slider("Max new tokens", min_value=40, max_value=220, value=128, step=8)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        st.divider()
        st.header("Safety")
        st.write("This prototype rewrites text for clarity. It does not diagnose, prescribe, or replace a care team.")

    if st.button("Rewrite", type="primary", disabled=not user_text.strip()):
        config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature)
        output = generate_rewrite(user_text, model, tokenizer, config)

        st.subheader("Plain-English Summary")
        st.write(output)
        render_jargon_table(user_text)

        left, right = st.columns(2)
        with left:
            render_metrics("Original text", user_text)
        with right:
            render_metrics("Generated rewrite", output)

    st.divider()
    st.subheader("What This Prototype Shows")
    st.write(
        "The fine-tuned adapter is trained on paired examples that map clinical wording to plain language. "
        "The demo combines generated simplification with deterministic glossary explanations for important "
        "medical terms."
    )


if __name__ == "__main__":
    main()
