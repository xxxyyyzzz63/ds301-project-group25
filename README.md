# AI-Generated Hotel Review Detector

**Course**: DS-UA 301 - Advanced Topics in Data Science (NYU Spring 2025)  
**Team**: Wendy, Wency, Yujia

## Overview
This project studies whether a stylometry-based, feature-driven detector can distinguish AI-generated hotel reviews from human-written reviews more reliably and transparently than a single-pass prompting-only LLM baseline. The final system is organized as a multi-step pipeline: a review is converted into stylometric features, scored by a trained classifier, calibrated into a 0 to 100 AI-likeness score, and paired with a short explanation grounded in the extracted features.

Across the project, we first built a prompting-only baseline, then trained stylometry-based classifiers, then added calibration and explanation generation, and finally updated the pipeline using a more diversified AI-generated review dataset. The final repository therefore preserves both the earlier milestone notebooks and the updated final detector artifacts.

## Research Question
Can a stylometry-based, feature-driven detector with structured outputs provide more reliable AI-vs-human hotel review detection and clearer evidence-based explanations than a single-pass prompting-only LLM baseline?

## Final Project Status
The repository now includes:

- the Week 2 prompting-only baseline
- the Week 3 stylometry-based classifier artifacts
- the integrated Week 4 explanation-chain style detector workflow
- the Week 5 evaluation and subgroup analysis
- an updated final detector based on a more diversified AI review dataset
- a reusable Python detector module for testing chat-style review inputs
- an end-to-end demo notebook

The diversified-data update made the task somewhat harder for logistic regression, but the final selected random forest model still performed extremely strongly, so results should still be interpreted with caution because residual synthetic cues may remain in the AI dataset.

## Repository Structure
```text
ds301-project-group25/
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── baseline_results.pkl
│   ├── lr_classifier.pkl
│   ├── lr_temp_scaler.pkl
│   ├── rf_classifier.pkl
│   ├── rf_temp_scaler.pkl
│   ├── week3_artifacts.pkl
│   ├── diverse_week5_artifacts.pkl
│   └── updated_week4_week5_outputs.pkl
│   ├── subgroup_analysis_outputs.pkl
│   └── dataset_audit_outputs.pkl
│   └── casing_ood_experiment_outputs.pkl
│
├── notebooks/
│   ├── AI_Review_Detector_Week1_3_Complete.ipynb
│   ├── ai_review_generation_and_eda.ipynb
│   ├── data_preparation.ipynb
│   ├── evaluate_baseline.ipynb
│   ├── subgroup_analysis_by_length.ipynb
│   ├── dataset_audit_and_shortcut_analysis.ipynb
│   ├── week4_week5_updated_with_diverse_data.ipynb
│   ├── detector_sanity_check.ipynb
│   ├── casing_ood_experiment.ipynb
│   └── final_demo.ipynb
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

## Setup Instructions:
1. Clone this repository.
2. Install dependencies: pip install -r requirements.txt
3. Download the dataset files separately and place them in your working directory:
- tripadvisor_hotel_reviews.csv
- ai_generated_tripadvisor_reviews_gemma3_4b.csv
4. Set your API key as an environment variable or notebook secret, depending on how you are running the code.

## Notebooks
- notebooks/ai_review_generation_and_eda.ipynb: Generates AI reviews and produces exploratory analysis used in the proposal-stage workflow.
- notebooks/AI_Review_Detector_Week1_3_Complete.ipynb: Main Weeks 1 to 3 implementation notebook, including feature engineering, classifier training, and saved model artifacts.
- notebooks/data_preparation.ipynb: Dataset cleaning and preprocessing notebook.
- notebooks/evaluate_baseline.ipynb: Week 2 prompting-only baseline evaluation notebook.
- notebooks/week4_week5_updated_with_diverse_data.ipynb: Updated Week 4 to 5 notebook that loads the final diversified-data artifact, reruns the explanation-chain demo, and reports updated evaluation and subgroup analysis.
- notebooks/detector_sanity_check.ipynb: Simple end-to-end detector loading and testing notebook for final integration checks.
- notebooks/subgroup_analysis_by_length.ipynb: Splits the test set into short/long reviews and reports per-subgroup classification metrics, calibration quality, and uncertain-band activation from pre-saved predictions.
- notebooks/dataset_audit_and_shortcut_analysis.ipynb: Audits train/val/test leakage, AI boilerplate prefixes, and probability saturation to responsibly interpret the perfect F1 score flagged in Milestone 2.
- notebooks/casing_ood_experiment.ipynb: Quantifies the casing OOD failure through three sub-experiments, lowercasing AI reviews, sentence-casing human reviews, and stepwise transformation attribution, to pinpoint which preprocessing operation drives the misclassification.
- notebooks/final_demo.ipynb: End-to-end demo running the final detector on seven curated reviews covering key edge cases.


## Source Code
- src/baseline_detector.py: Prompting-only baseline logic from the earlier project stage.
- src/stylometry_features.py: Stylometric feature extraction utilities.
- src/final_detector.py: Final detector interface that loads the saved artifact and returns structured prediction outputs.

## Model Artifacts
The repository keeps both earlier and final artifacts:
- earlier milestone artifacts are preserved to document project development
- diverse_week5_artifacts.pkl is the main final artifact for the updated detector
- updated_week4_week5_outputs.pkl stores the updated Week 4 to 5 evaluation outputs
- subgroup_analysis_outputs.pkl stores the saved tables (per-subgroup metrics, band activation, calibration quality, errors) from the subgroup analysis notebook
- dataset_audit_outputs.pkl stores the leakage, boilerplate, saturation, and per-opening tables from the audit notebook
- casing_ood_experiment_outputs.pkl stores the per-review predictions for Experiments A, B and the stepwise attribution table from Experiment C.

## References
See the project proposal and milestone materials for the full methodology, literature review, planned pipeline design, and evaluation rationale.

## Notes

- Large CSV data files are not uploaded to GitHub.
- Model .pkl files are included because later stages of the project load trained classifiers and calibration artifacts directly.
- Earlier notebooks reflect the original synthetic AI dataset workflow.
- The updated final detector uses a more diversified AI review dataset created to reduce repetitive generation patterns and make evaluation more realistic.
- Even in the updated workflow, results should still be interpreted cautiously because synthetic AI reviews may retain residual cues that make them easier to separate than real-world mixed-origin text.
