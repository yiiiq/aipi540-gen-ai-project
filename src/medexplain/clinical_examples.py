"""Curated clinical-note examples and jargon explanations for the demo app."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class JargonDefinition:
    """One clinical term and its patient-friendly meaning."""

    term: str
    definition: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClinicalExample:
    """One curated demo note with a safe summary and glossary."""

    title: str
    source_text: str
    plain_english: str
    jargon: tuple[JargonDefinition, ...]


CLINICAL_EXAMPLES: tuple[ClinicalExample, ...] = (
    ClinicalExample(
        title="Emergency Department - possible heart attack",
        source_text=(
            "56-year-old male presents with acute onset substernal chest pain radiating to the left arm. "
            "ECG demonstrates ST-segment elevation in the inferior leads. Troponin I markedly elevated. "
            "Assessment consistent with STEMI. Cardiology consulted for emergent PCI."
        ),
        plain_english=(
            "The patient likely has a heart attack and needs an emergency procedure to open the blocked artery."
        ),
        jargon=(
            JargonDefinition("ECG", "A heart rhythm test, also called an electrocardiogram.", ("EKG",)),
            JargonDefinition("ST-segment elevation", "A heart-test pattern that can signal a heart attack."),
            JargonDefinition("Troponin I", "A blood marker that rises when the heart muscle is injured."),
            JargonDefinition("STEMI", "A serious type of heart attack.", ("ST elevation myocardial infarction",)),
            JargonDefinition("PCI", "A procedure to open a blocked heart artery.", ("angioplasty",)),
        ),
    ),
    ClinicalExample(
        title="Internal Medicine - possible heart failure flare",
        source_text=(
            "Patient denies fever, chills, nausea, vomiting, or diarrhea. Physical exam notable for bilateral "
            "lower extremity pitting edema and bibasilar crackles. Findings concerning for acute decompensated CHF."
        ),
        plain_english=(
            "The patient does not report infection or stomach symptoms, but has leg swelling and lung sounds "
            "that may mean worsening congestive heart failure."
        ),
        jargon=(
            JargonDefinition("Denies", "Reports not having."),
            JargonDefinition("Pitting edema", "Swelling that leaves an indentation when pressed."),
            JargonDefinition("Bibasilar crackles", "Crackling sounds heard at the bases of the lungs."),
            JargonDefinition("CHF", "Congestive heart failure.", ("congestive heart failure",)),
        ),
    ),
    ClinicalExample(
        title="Orthopedics - ACL and meniscus injury",
        source_text=(
            "MRI reveals full-thickness ACL tear with associated medial meniscus injury. Recommend arthroscopic "
            "ACL reconstruction following resolution of acute inflammation."
        ),
        plain_english=(
            "The knee MRI shows a complete ACL tear and an injury to the inner knee cartilage. Surgery is "
            "recommended after the swelling and irritation calm down."
        ),
        jargon=(
            JargonDefinition("MRI", "An imaging test that uses magnets to show detailed pictures inside the body."),
            JargonDefinition("ACL", "A major ligament that helps stabilize the knee."),
            JargonDefinition("Medial meniscus", "Cartilage on the inner side of the knee that cushions the joint."),
            JargonDefinition("Arthroscopic", "Done with small instruments and a camera through small cuts."),
            JargonDefinition("Reconstruction", "Surgery to rebuild or replace a damaged structure."),
        ),
    ),
    ClinicalExample(
        title="Neurology - possible stroke",
        source_text=(
            "Patient presents with dysarthria and right-sided hemiparesis. Non-contrast CT negative for "
            "intracranial hemorrhage. Symptoms highly suspicious for acute ischemic CVA."
        ),
        plain_english=(
            "The patient has slurred speech and weakness on the right side. The scan does not show bleeding "
            "in the brain, so the symptoms are concerning for a stroke caused by blocked blood flow."
        ),
        jargon=(
            JargonDefinition("Dysarthria", "Slurred or difficult speech."),
            JargonDefinition("Hemiparesis", "Weakness on one side of the body."),
            JargonDefinition("Non-contrast CT", "A CT scan done without contrast dye."),
            JargonDefinition("Intracranial hemorrhage", "Bleeding inside the brain or skull."),
            JargonDefinition("CVA", "Stroke.", ("cerebrovascular accident",)),
        ),
    ),
    ClinicalExample(
        title="Cardiology - AFib medication plan",
        source_text=(
            "Assessment: Persistent atrial fibrillation with rapid ventricular response. CHA2DS2-VASc score "
            "of 4. Continue apixaban for anticoagulation and initiate metoprolol for rate control."
        ),
        plain_english=(
            "The patient has ongoing irregular heartbeat with a fast heart rate. The plan is to continue a "
            "blood thinner to lower stroke risk and start medicine to slow the heart rate."
        ),
        jargon=(
            JargonDefinition("Atrial fibrillation", "An irregular heartbeat, often called AFib.", ("AFib",)),
            JargonDefinition("Rapid ventricular response", "A fast heart rate caused by the irregular rhythm.", ("RVR",)),
            JargonDefinition("CHA2DS2-VASc score", "A score doctors use to estimate stroke risk in AFib."),
            JargonDefinition("Anticoagulation", "Blood-thinning treatment."),
            JargonDefinition("Rate control", "Treatment to slow the heart rate."),
        ),
    ),
    ClinicalExample(
        title="Pulmonology - abnormal chest CT",
        source_text=(
            "CT chest demonstrates multifocal ground-glass opacities bilaterally without pleural effusion. "
            "Differential includes atypical pneumonia versus organizing pneumonitis."
        ),
        plain_english=(
            "The chest CT shows hazy spots in both lungs, but no extra fluid around the lungs. Possible causes "
            "include an unusual pneumonia or inflammation in the lung tissue."
        ),
        jargon=(
            JargonDefinition("CT chest", "A detailed X-ray scan of the chest."),
            JargonDefinition("Ground-glass opacities", "Hazy-looking areas on a lung scan."),
            JargonDefinition("Pleural effusion", "Extra fluid around the lungs."),
            JargonDefinition("Differential", "The list of possible causes doctors are considering.", ("differential diagnosis",)),
            JargonDefinition("Organizing pneumonitis", "Inflammation and healing changes in the lungs."),
        ),
    ),
    ClinicalExample(
        title="Gastroenterology - stomach inflammation",
        source_text=(
            "EGD demonstrated erosive gastritis without evidence of active upper GI bleeding. Biopsies "
            "obtained to evaluate for Helicobacter pylori."
        ),
        plain_english=(
            "The upper endoscopy showed irritation and damage in the stomach lining, but no active bleeding. "
            "Small tissue samples were taken to check for H. pylori infection."
        ),
        jargon=(
            JargonDefinition("EGD", "Upper endoscopy, a camera test of the esophagus, stomach, and first part of the small intestine.", ("upper endoscopy",)),
            JargonDefinition("Erosive gastritis", "Irritation and wearing away of the stomach lining."),
            JargonDefinition("GI bleeding", "Bleeding in the digestive tract."),
            JargonDefinition("Biopsy", "A small tissue sample taken for testing.", ("biopsies",)),
            JargonDefinition("H. pylori", "A bacteria that can cause stomach irritation or ulcers.", ("helicobacter pylori",)),
        ),
    ),
    ClinicalExample(
        title="Oncology - cancer treatment planning",
        source_text=(
            "PET/CT demonstrates interval progression of metastatic disease with new hepatic lesions. "
            "Recommend initiation of second-line systemic chemotherapy pending molecular profiling."
        ),
        plain_english=(
            "The scan shows the cancer has grown or spread, with new spots in the liver. The team recommends "
            "another chemotherapy treatment while waiting for tumor testing results."
        ),
        jargon=(
            JargonDefinition("PET/CT", "A scan that combines metabolism imaging and detailed body imaging."),
            JargonDefinition("Metastatic disease", "Cancer that has spread from where it started."),
            JargonDefinition("Hepatic lesions", "Abnormal spots in the liver."),
            JargonDefinition("Systemic chemotherapy", "Cancer medicine that travels through the whole body."),
            JargonDefinition("Molecular profiling", "Testing tumor genes or markers to guide treatment."),
        ),
    ),
    ClinicalExample(
        title="Infectious Disease - bloodstream infection",
        source_text=(
            "Blood cultures positive for MRSA bacteremia. Transthoracic echocardiogram ordered to rule out "
            "infective endocarditis. Continue IV vancomycin."
        ),
        plain_english=(
            "The blood test shows MRSA bacteria in the bloodstream. The team ordered a heart ultrasound to "
            "check for infection on the heart valves and will continue IV antibiotics."
        ),
        jargon=(
            JargonDefinition("MRSA", "A type of staph bacteria resistant to some common antibiotics."),
            JargonDefinition("Bacteremia", "Bacteria in the bloodstream."),
            JargonDefinition("Echocardiogram", "An ultrasound test of the heart."),
            JargonDefinition("Infective endocarditis", "An infection of the heart lining or valves."),
            JargonDefinition("IV vancomycin", "An antibiotic given through a vein."),
        ),
    ),
    ClinicalExample(
        title="ICU - ventilator progress note",
        source_text=(
            "Patient remains intubated and mechanically ventilated. Hemodynamically stable on low-dose "
            "norepinephrine. Urine output adequate. Continue sedation wean and spontaneous breathing trial "
            "tomorrow morning."
        ),
        plain_english=(
            "The patient still has a breathing tube and breathing machine. Blood pressure is stable with a "
            "small amount of support medicine, urine output is okay, and the team plans to reduce sedation "
            "and test breathing without full machine support tomorrow."
        ),
        jargon=(
            JargonDefinition("Intubated", "Has a breathing tube in place."),
            JargonDefinition("Mechanically ventilated", "Supported by a breathing machine."),
            JargonDefinition("Hemodynamically stable", "Blood pressure and circulation are stable."),
            JargonDefinition("Norepinephrine", "Medicine that helps support blood pressure."),
            JargonDefinition("Sedation wean", "Slowly reducing medicines that keep a patient sleepy."),
            JargonDefinition("Spontaneous breathing trial", "A test to see if the patient can breathe with less machine support."),
        ),
    ),
)


def normalize_text(text: str) -> str:
    """Normalize text for matching typed examples and glossary aliases."""

    normalized = text.lower().replace("₂", "2").replace("₂", "2")
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def find_matching_example(text: str) -> ClinicalExample | None:
    """Return the curated example when the user input matches one exactly."""

    normalized = normalize_text(text)
    for example in CLINICAL_EXAMPLES:
        if normalize_text(example.source_text) == normalized:
            return example
    return None


def all_jargon_definitions() -> tuple[JargonDefinition, ...]:
    """Return unique glossary definitions across examples."""

    definitions: dict[str, JargonDefinition] = {}
    for example in CLINICAL_EXAMPLES:
        for item in example.jargon:
            definitions.setdefault(normalize_text(item.term), item)
    return tuple(definitions.values())


def extract_jargon(text: str) -> tuple[JargonDefinition, ...]:
    """Find known clinical jargon terms in custom text."""

    normalized = normalize_text(text)
    found: list[JargonDefinition] = []
    for item in all_jargon_definitions():
        candidates = (item.term, *item.aliases)
        if any(normalize_text(candidate) in normalized for candidate in candidates):
            found.append(item)
    return tuple(found)
