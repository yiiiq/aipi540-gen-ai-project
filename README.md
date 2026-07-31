# MedExplain

MedExplain fine-tunes a small instruction model to rewrite clinical language into patient-friendly explanations. The project is designed for Mini Hackathon #4: it adapts a generative model, shows a before/after comparison, and deploys a simple app that runs inference with the trained adapter.

AI assistance disclosure: this project scaffold and code were created with help from OpenAI Codex. See [AI_USAGE.md](AI_USAGE.md).

## Project Goal

Patients often receive visit notes, lab explanations, and discharge instructions that are technically correct but difficult to understand. MedExplain gives a machine a targeted new capability: rewriting medical jargon into plain language while preserving the original meaning and avoiding unsupported advice.

## Model And Adaptation Strategy

- Base model: `google/flan-t5-small`
- Adaptation method: LoRA with PEFT
- Task format: text-to-text generation
- Input: clinical or medical text
- Output: patient-friendly rewrite
- Main training data: Med-EASi (`cbasu/Med-EASi`) expert-to-simple medical text pairs

The trained adapter is saved under `models/medexplain-lora/`. The Streamlit app loads that adapter when it is present.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── setup.py
├── main.py
├── scripts
│   ├── make_dataset.py
│   ├── train_model.py
│   └── evaluate_model.py
├── src/medexplain
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── model.py
│   └── prompts.py
├── data
│   ├── raw
│   ├── processed
│   └── outputs
├── models
├── notebooks
└── tests
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/make_dataset.py
python scripts/train_model.py
python scripts/evaluate_model.py
streamlit run main.py
```

By default, `scripts/make_dataset.py` downloads Med-EASi from Hugging Face and keeps the first 100 rows for a fast hackathon training run. This final project adapter was trained on that 100-row Med-EASi subset for 40 epochs so the live demo shows a visible learned rewrite behavior. To use the original 20-row seed dataset instead:

```bash
python scripts/make_dataset.py --source seed
```

## Deploy

The easiest deployment path is Streamlit Community Cloud.

1. Push this repository to GitHub.
2. Run `python scripts/train_model.py` locally.
3. Commit the lightweight LoRA adapter in `models/medexplain-lora/`, or upload it to Hugging Face and set `MEDEXPLAIN_MODEL_DIR` to that adapter path.
4. In Streamlit Cloud, choose `main.py` as the app entrypoint.

If the adapter directory is missing, the app still runs with the base model and clearly marks that the fine-tuned adapter is not active. For the final submission, deploy with the adapter available so the app satisfies the trained-model inference requirement.

## Evaluation

The evaluation script creates `data/outputs/before_after_examples.csv` with:

- base model output
- fine-tuned model output
- plain-language readability metrics
- jargon term counts

## Responsible Use

MedExplain is an educational prototype. It should not be used for diagnosis, treatment decisions, or replacing a clinician. A production version would need expert-reviewed data, safety filters, clinical validation, and clear uncertainty handling.

## Data And Code Attribution

- Med-EASi dataset: https://huggingface.co/datasets/cbasu/Med-EASi
- Med-EASi paper listed on Hugging Face: arXiv:2302.09155
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- Hugging Face PEFT documentation: https://huggingface.co/docs/peft
- AI assistance: project code and documentation were drafted with OpenAI Codex and reviewed/modified in this repository.
