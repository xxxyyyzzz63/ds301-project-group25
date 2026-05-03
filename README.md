# AI-Generated Hotel Review Detector

**Course**: DS-UA 301 - Advanced Topics in Data Science (NYU Spring 2025)
**Team**: Wendy, Wency, Yujia

## Overview
This project studies whether a stylometry-based, feature-driven detector can distinguish AI-generated hotel reviews from human-written reviews more reliably and transparently than a single-pass prompting-only LLM baseline. The final system is organised as a multi-step pipeline: a review is converted into stylometric features, scored by a trained classifier, calibrated into a 0 to 100 AI-likeness score, and paired with a short explanation grounded in the extracted features.

Across the project, we first built a prompting-only baseline, then trained stylometry-based classifiers, then added calibration and explanation generation, and finally updated the pipeline using a more diversified AI-generated review dataset. The Milestone 3 wrap-up adds three robustness analyses (subgroup-by-length, dataset audit, hybrid stress test), a reusable detector module, and an end-to-end demo notebook.

## Research Question
Can a stylometry-based, feature-driven detector with structured outputs provide more reliable AI-vs-human hotel review detection and clearer evidence-based explanations than a single-pass prompting-only LLM baseline?

## Final Project Status
The repository now includes:

- the Week 2 prompting-only baseline
- the Week 3 stylometry-based classifier artifacts
- the integrated Week 4 explanation-chain style detector workflow (LangChain `RunnableLambda` chain)
- the Week 5 evaluation and subgroup analysis
- an updated final detector trained on a more diversified AI-generated review dataset
- a dataset audit, a clean retrain on a label-wise deduplicated split, and a hybrid stylometry + TF-IDF stress test
- a reusable `FinalReviewDetector` Python class for testing chat-style review inputs
- an end-to-end demo notebook (`notebooks/final_demo.ipynb`)

The final selected Random Forest model achieves F1 = 1.000 on the held-out test split. The Milestone 3 audit shows this number reflects both real stylometric signal and residual distribution-level cues in the synthetic AI dataset that survive deduplication and a generator change. Hand-written human-style sanity-check reviews written in normal sentence-cased English are classified as AI with probability ≈ 1.0, which suggests the detector partly cues on capitalisation and stopword density patterns that separate the lowercased TripAdvisor human dataset from our AI dataset, rather than on a generator-agnostic AI-style signature. Results are reported as promising rather than evidence of cross-distribution robustness.

## Quick start: run the demo
```bash
# 1) Clone and install
pip install -r requirements.txt

# 2) Run the end-to-end demo
jupyter notebook notebooks/final_demo.ipynb

# OR use the detector directly from Python
python -c "
from src.final_detector import FinalReviewDetector
from src.stylometry_features import extract_stylometry_features
detector = FinalReviewDetector()
text = 'The hotel was fine. Clean room, decent service.'
features = extract_stylometry_features(text)
print(detector.detect_review_dict(review_text=text, extracted_features=features))
"

# OR run the smoke test
python test_final_detector.py
```

## Repository Structure
```text
ds301-project-group25/
├── README.md
├── requirements.txt
├── test_final_detector.py
├── .gitignore
│
├── models/
│   ├── ai_detector_tools.pkl
│   ├── baseline_results.pkl
│   ├── lr_classifier.pkl
│   ├── lr_temp_scaler.pkl
│   ├── rf_classifier.pkl
│   ├── rf_temp_scaler.pkl
│   ├── week3_artifacts.pkl
│   ├── diverse_week5_artifacts.pkl
│   ├── updated_week4_week5_outputs.pkl
│   ├── subgroup_analysis_outputs.pkl
│   ├── dataset_audit_outputs.pkl
│   ├── week5_audit_outputs.pkl
│   ├── week5_outputs.pkl
│   └── week4_explanation_chain_metadata.pkl
│
├── notebooks/
│   ├── AI_Review_Detector_Week1_3_Complete.ipynb
│   ├── ai_review_generation_and_eda.ipynb
│   ├── data_preparation.ipynb
│   ├── evaluate_baseline.ipynb
│   ├── week4_week5_explanation_chain_and_audit.ipynb
│   ├── week4_week5_updated_with_diverse_data.ipynb
│   ├── subgroup_analysis_by_length.ipynb
│   ├── dataset_audit_and_shortcut_analysis.ipynb
│   ├── detector_sanity_check.ipynb
│   ├── final_demo.ipynb
│   └── README.md
│
├── src/
│   ├── baseline_detector.py
│   ├── stylometry_features.py
│   └── final_detector.py
│
└── Data files (not uploaded to GitHub):
    ├── tripadvisor_hotel_reviews.csv
    ├── ai_generated_tripadvisor_reviews_gemma3_4b.csv
    └── ai_generated_tripadvisor_reviews_openai_diverse.csv
```

## Setup Instructions
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset files separately and place them in your working directory:
   - `tripadvisor_hotel_reviews.csv`
   - `ai_generated_tripadvisor_reviews_gemma3_4b.csv`
   - `ai_generated_tripadvisor_reviews_openai_diverse.csv` (used by the diversified-data update)
4. Set your API key as an environment variable or notebook secret if you want to re-run the prompting-only baseline or the LLM-grounded explanation step.

## Notebooks
- `notebooks/ai_review_generation_and_eda.ipynb`: Generates AI reviews and produces exploratory analysis used in the proposal-stage workflow.
- `notebooks/AI_Review_Detector_Week1_3_Complete.ipynb`: Main Weeks 1 to 3 implementation notebook, including feature engineering, classifier training, calibration, and saved model artifacts.
- `notebooks/data_preparation.ipynb`: Dataset cleaning and preprocessing notebook.
- `notebooks/evaluate_baseline.ipynb`: Week 2 prompting-only baseline evaluation notebook.
- `notebooks/week4_week5_explanation_chain_and_audit.ipynb`: Week 4 to 5 notebook combining the LangChain runnable explanation chain with the original-data Week 5 evaluation, the leakage / duplication / boilerplate audit, the clean retrain on the deduplicated split, and the hybrid stylometry + character/word TF-IDF stress test.
- `notebooks/week4_week5_updated_with_diverse_data.ipynb`: Updated Week 4 to 5 notebook that loads the diversified-data artifact, reruns the explanation-chain demo, and reports updated evaluation and subgroup analysis.
- `notebooks/subgroup_analysis_by_length.ipynb`: Subgroup analysis splitting the test set into short (<79 words) and long (≥79 words) reviews. Reports per-subgroup accuracy, precision, recall, F1, confusion matrices, score separation plots, calibration quality (Brier score, log loss), and uncertain-band activation. Loads predictions from `updated_week4_week5_outputs.pkl` so no retraining is needed.
- `notebooks/dataset_audit_and_shortcut_analysis.ipynb`: Quantifies train/val/test leakage, generator-style boilerplate prefixes in the AI dataset, calibrated-probability saturation, and per-opening detection accuracy. Synthesises the findings to interpret the perfect F1 score responsibly.
- `notebooks/detector_sanity_check.ipynb`: Simple end-to-end detector loading and testing notebook for final integration checks.
- `notebooks/final_demo.ipynb`: Cleaned end-to-end demo of the final detector on seven curated reviews — clear AI, AI with prompt echo, short AI, lowercased human, long sentence-cased human, ambiguous mixed, and the OOD failure case from the sanity check. Loads `src/final_detector.py` directly and works whether you launch jupyter from the repo root or from `notebooks/`.

## Source Code
- `src/baseline_detector.py`: Prompting-only baseline detector. LangChain `LLMChain` over a few-shot chain-of-thought prompt; exposes `BaselineDetector` and an `AIDetectionResult` Pydantic schema.
- `src/stylometry_features.py`: Stylometric feature extractor. Single function `extract_stylometry_features(text)` returning the 14-feature dict in the canonical column order expected by the trained classifiers.
- `src/final_detector.py`: Final detector class. `FinalReviewDetector` loads `models/diverse_week5_artifacts.pkl` (random forest + temperature scaler trained on the diversified-data split) and exposes:
    - `detect_review(review_text, extracted_features)` returning a `FinalDetectionOutput` Pydantic object
    - `detect_review_dict(review_text, extracted_features)` returning a plain dict
    - helper methods `run_classifier_on_features`, `apply_temperature_scaling`, `get_uncertainty_band`, `get_top_features_for_explanation`, `generate_grounded_explanation`

  The output dict fields: `ai_probability`, `ai_likeness_score`, `uncertainty_band`, `predicted_label`, `top_features`, `explanation`.

## Model Artifacts
The repository keeps both earlier and final artifacts:
- Earlier milestone artifacts are preserved to document project development.
- `diverse_week5_artifacts.pkl` is the main final artifact loaded by `final_detector.py` (Random Forest + temperature scaler trained on the diversified-data split).
- `updated_week4_week5_outputs.pkl` stores the updated Week 4 to 5 evaluation outputs (full test predictions, subgroup metrics, uncertainty-band breakdown).
- `subgroup_analysis_outputs.pkl` stores the per-subgroup metrics, band activation, calibration quality, and error tables from the subgroup analysis notebook.
- `dataset_audit_outputs.pkl` stores the leakage, boilerplate, saturation, and per-opening tables from the audit notebook.
- `week5_audit_outputs.pkl` stores the full split-overlap counts, opening-phrase frequency table, dedup metrics, and dedup-test predictions referenced by the audit notebook.

## Headline Numbers
- Week 2 prompting-only baseline (100-review sample): F1 = 0.350
- Final calibrated Random Forest on diversified data (full 5,185-review test set): F1 = 1.000
- F1 improvement over baseline: +0.650 (well above the ≥0.05 success criterion in the proposal)
- Subgroup F1 (short / long): 1.000 / 1.000; uncertain-band activation: 1 of 5,185 reviews
- Probability saturation on test set: 99.98% of predictions sit at <0.01 or >0.99
- Dominant AI-dataset opening: a single 8-word phrase covers 26.61% of all AI training reviews
- Label-wise deduplication: AI dataset shrinks 10,000 → 867 unique reviews; F1 stays at 1.000
- Hybrid stylometry + TF-IDF stress test on the deduplicated split: F1 = 1.000
- Hand-written human-style sanity-check reviews written in normal sentence-cased English are classified as AI with probability ≈ 1.0 (out-of-distribution failure mode)

## References
See the project proposal and milestone materials for the full methodology, literature review, planned pipeline design, and evaluation rationale.

## Notes

- Large CSV data files are not uploaded to GitHub.
- Model `.pkl` files are included because later stages of the project load trained classifiers and calibration artifacts directly.
- Earlier notebooks reflect the original Gemma-3 synthetic AI dataset workflow.
- The updated final detector uses a more diversified OpenAI-generated AI review dataset created to reduce repetitive generation patterns and make evaluation more realistic.
- Even in the updated workflow, results should still be interpreted cautiously because synthetic AI reviews may retain residual distribution-level cues that make them easier to separate than mixed-origin real-world text. The hand-written sanity-check failures discussed in Milestone 3 are the clearest evidence of this caveat.
