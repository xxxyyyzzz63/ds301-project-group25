# Hybrid AI-Generated Hotel Review Auditing Framework

**Course**: DS-UA 301 - Advanced Topics in Data Science (NYU Spring 2025)  
**Team**: Wendy, Wency, Yujia

# Overview
This project studies whether a hybrid multi-stage framework can distinguish AI-generated hotel reviews from human-written reviews more reliably and transparently than a single-pass prompting-only baseline.

The final system combines:

- LLM-based linguistic analysis
- stylometric feature extraction
- statistical machine learning classification
- fusion-based adjudication
- robustness and attribution analysis

Rather than acting as a simple binary detector, the framework is designed to provide:

- interpretable reasoning
- calibrated AI-likeness scoring
- uncertainty estimation
- disagreement analysis
- robustness validation
- dataset shortcut auditing

The repository preserves both:

- earlier milestone notebooks documenting project evolution
- the redesigned final framework implemented as reusable Python modules and scripts

# Research Question
Can a hybrid multi-stage framework combining LLM-based linguistic reasoning and stylometric machine learning provide more reliable and interpretable AI-vs-human hotel review detection than a prompting-only baseline?

# System Architecture 

The final framework is organized into three layers. 

## Layer 1: Primary Pipeline 

Review 
→ LLM Linguistic Analyzer 
→ Stylometric ML Classifier 
→ Fusion Adjudicator 
→ Final Decision 

This layer combines semantic reasoning with statistical stylometric detection. 

---

## Layer 2: Analysis & Validation 

The framework includes several analysis components designed to evaluate reliability and interpretability: 

- Attribution Analyzer 
Examines which stylometric features most strongly influence predictions. 

- Disagreement Analyzer 
Identifies cases where different system components disagree. 

- Component Validator 
Evaluates consistency and reliability of LLM-based analysis modules. 

- Benchmark Suite 
Compares the hybrid system against simpler baselines. 

- Prompt Framework Documentation
 Documents prompt structures and reasoning workflows used in the LLM stages. 

---

## Layer 3: Integration 
The final layer integrates: 

- reusable scripts 
- saved model artifacts 
- evaluation outputs 
- demo workflows 
- analysis notebooks 

into a unified experimental framework.



# Final Project Status

 The repository now includes: 
 
 - the original prompting-only baseline detector 
 - stylometry-based ML classifiers 
 - calibrated AI-likeness scoring 
 - explanation-chain workflows 
 - diversified AI review datasets 
 - reusable modular Python components 
 - robustness and audit notebooks 
 - disagreement and attribution analysis 
 - stress testing and failure-case analysis 
 - benchmark and validation scripts 
 
 The diversified-data update made the classification task substantially harder for logistic regression, while the final selected random forest model still performed extremely strongly. 
 
 However, additional robustness analysis revealed that some human-written reviews can still be misclassified as strongly AI-generated, suggesting that residual stylometric shortcuts or synthetic dataset artifacts may remain.

# Hybrid AI-Generated Hotel Review Auditing Framework

**Course**: DS-UA 301 - Advanced Topics in Data Science (NYU Spring 2025)
**Team**: Wendy, Wency, Yujia

---

# Overview

This project studies whether a hybrid multi-stage framework can distinguish AI-generated hotel reviews from human-written reviews more reliably and transparently than a single-pass prompting-only baseline.

The final system combines:

* LLM-based linguistic analysis
* stylometric feature extraction
* statistical machine learning classification
* fusion-based adjudication
* robustness and attribution analysis

Rather than acting as a simple binary detector, the framework is designed to provide:

* interpretable reasoning
* calibrated AI-likeness scoring
* uncertainty estimation
* disagreement analysis
* robustness validation
* dataset shortcut auditing

The repository preserves both:

* earlier milestone notebooks documenting project evolution
* the redesigned final framework implemented as reusable Python modules and scripts

---

# Research Question

Can a hybrid multi-stage framework combining LLM-based linguistic reasoning and stylometric machine learning provide more reliable and interpretable AI-vs-human hotel review detection than a prompting-only baseline?

---

# System Architecture

The final framework is organized into three layers.

## Layer 1: Primary Pipeline

Review
→ LLM Linguistic Analyzer
→ Stylometric ML Classifier
→ Fusion Adjudicator
→ Final Decision

This layer combines semantic reasoning with statistical stylometric detection.

---

## Layer 2: Analysis & Validation

The framework includes several analysis components designed to evaluate reliability and interpretability:

* Attribution Analyzer
  Examines which stylometric features most strongly influence predictions.

* Disagreement Analyzer
  Identifies cases where different system components disagree.

* Component Validator
  Evaluates consistency and reliability of LLM-based analysis modules.

* Benchmark Suite
  Compares the hybrid system against simpler baselines.

* Prompt Framework Documentation
  Documents prompt structures and reasoning workflows used in the LLM stages.

---

## Layer 3: Integration

The final layer integrates:

* reusable scripts
* saved model artifacts
* evaluation outputs
* demo workflows
* analysis notebooks

into a unified experimental framework.

---

# Final Project Status

The repository now includes:

* the original prompting-only baseline detector
* stylometry-based ML classifiers
* calibrated AI-likeness scoring
* explanation-chain workflows
* diversified AI review datasets
* reusable modular Python components
* robustness and audit notebooks
* disagreement and attribution analysis
* stress testing and failure-case analysis
* benchmark and validation scripts

The diversified-data update made the classification task substantially harder for logistic regression, while the final selected random forest model still performed extremely strongly.

However, additional robustness analysis revealed that some human-written reviews can still be misclassified as strongly AI-generated, suggesting that residual stylometric shortcuts or synthetic dataset artifacts may remain.

---

# Repository Structure

```text
ds301-project-group25/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── ai_generated_tripadvisor_reviews_openai_diverse.csv
│
├── models/
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
│   ├── casing_ood_experiment_outputs.pkl
│   ├── week4_explanation_chain_metadata.pkl
│   ├── week5_outputs.pkl
│   └── week5_audit_outputs.pkl
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
│   ├── casing_ood_experiment.ipynb
│   ├── borderline_analysis.ipynb
│   ├── stress_test.ipynb
│   ├── detector_sanity_check.ipynb
│   └── final_demo.ipynb
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── llm_linguistic_analyzer.py
│   ├── stylometry_features.py
│   ├── final_detector.py
│   ├── fusion_adjudicator.py
│   ├── attribution_analyzer.py
│   ├── disagreement_analyzer.py
│   ├── llm_component_validator.py
│   ├── benchmark.py
│   ├── baseline_detector.py
│   ├── prompt_engineering_docs.py
│   ├── test_ai_and_human.py
│   └── test_final_detector.py
│
├── scripts/
│   ├── run_demo.py
│   ├── run_benchmark.py
│   ├── run_attribution_analysis.py
│   ├── run_disagreement_analysis.py
│   ├── run_component_validation.py
│   └── run_prompt_framework.py
│
└── External data files (not uploaded to GitHub):
    └── tripadvisor_hotel_reviews.csv
```

---

# Setup Instructions

1. Clone the repository.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the external dataset separately and place it in the project root:

* `tripadvisor_hotel_reviews.csv`

4. Configure your OpenAI API key:

For notebook usage:

* set the key through Colab secrets or environment variables

For local usage:

* create an environment variable named:

```bash
OPENAI_API_KEY
```

---

# Core Pipeline Components

## Primary Pipeline

### `src/pipeline.py`

Main orchestration pipeline connecting all framework stages.

### `src/llm_linguistic_analyzer.py`

Performs semantic and linguistic analysis using an LLM.

### `src/stylometry_features.py`

Extracts stylometric and statistical writing features from reviews.

### `src/final_detector.py`

Loads trained classifiers and calibration artifacts to produce structured predictions.

### `src/fusion_adjudicator.py`

Combines outputs from semantic analysis and stylometric classification into a final decision.

---

# Analysis & Validation Modules

### `src/attribution_analyzer.py`

Analyzes which features contribute most strongly to predictions.

### `src/disagreement_analyzer.py`

Studies disagreement and instability between different pipeline components.

### `src/llm_component_validator.py`

Validates consistency and reliability of LLM-generated outputs.

### `src/benchmark.py`

Runs benchmark comparisons against simpler baselines.

### `src/prompt_engineering_docs.py`

Documents and organizes prompt frameworks used throughout the system.

---

# Supporting Modules

### `src/baseline_detector.py`

Earlier prompting-only baseline system.

### `src/test_final_detector.py`

Basic detector integration tests.

### `src/test_ai_and_human.py`

Runs curated AI/human review examples through the final pipeline.

---

# Scripts

### `scripts/run_demo.py`

Runs the end-to-end framework demo.

### `scripts/run_benchmark.py`

Executes benchmark comparisons across systems.

### `scripts/run_attribution_analysis.py`

Runs attribution analysis workflows.

### `scripts/run_disagreement_analysis.py`

Runs disagreement and failure-case analysis.

### `scripts/run_component_validation.py`

Evaluates LLM component reliability.

### `scripts/run_prompt_framework.py`

Runs prompt-engineering framework demonstrations.

---

# Notebooks

## Development & Training

* `AI_Review_Detector_Week1_3_Complete.ipynb`
  Weeks 1–3 implementation notebook containing feature engineering, model training, calibration, and saved artifacts.

* `data_preparation.ipynb`
  Dataset cleaning and preprocessing workflow.

* `ai_review_generation_and_eda.ipynb`
  AI review generation and exploratory analysis notebook.

---

## Evaluation & Analysis

* `week4_week5_explanation_chain_and_audit.ipynb`
  Integrated explanation-chain workflow and auditing experiments.

* `week4_week5_updated_with_diverse_data.ipynb`
  Final diversified-data evaluation notebook.

* `subgroup_analysis_by_length.ipynb`
  Performance analysis across review-length subgroups.

* `dataset_audit_and_shortcut_analysis.ipynb`
  Investigates dataset leakage, shortcut learning, and probability saturation.

* `borderline_analysis.ipynb`
  Examines predictions near the decision boundary and compares them against highly confident predictions.

* `stress_test.ipynb`
  Manual robustness testing using curated edge-case hotel reviews, including short, polished, messy, ambiguous, and neutral examples.

* `casing_ood_experiment.ipynb`
  Studies casing-based out-of-distribution failures and preprocessing sensitivity.

---

## Demo & Integration

* `detector_sanity_check.ipynb`
  Simple end-to-end loading and integration checks.

* `final_demo.ipynb`
  End-to-end framework demo using curated review examples.

---

# Model Artifacts

The repository preserves both earlier milestone artifacts and updated final-system artifacts.

Key files include:

* `diverse_week5_artifacts.pkl`
  Main final detector artifact.

* `updated_week4_week5_outputs.pkl`
  Saved predictions and evaluation outputs for the updated final system.

* `subgroup_analysis_outputs.pkl`
  Saved subgroup evaluation tables and uncertainty metrics.

* `dataset_audit_outputs.pkl`
  Outputs from leakage and shortcut-learning audits.

* `casing_ood_experiment_outputs.pkl`
  Outputs from casing sensitivity experiments.

---

# Source Code
- src/baseline_detector.py: Prompting-only baseline logic from the earlier project stage.
- src/stylometry_features.py: Stylometric feature extraction utilities.
- src/final_detector.py: Final detector interface that loads the saved artifact and returns structured prediction outputs.

---

# Robustness & Limitations

Although the final random forest detector achieves near-perfect performance on the current evaluation dataset, additional robustness analysis revealed several important limitations:

* polished human-written reviews can sometimes be classified as strongly AI-generated
* confidence calibration remains imperfect near the decision boundary
* stylometric shortcuts may still influence predictions
* synthetic AI datasets may retain residual artifacts

To better understand these issues, the repository includes:

* stress testing
* subgroup analysis
* borderline prediction analysis
* disagreement analysis
* shortcut auditing
* OOD casing experiments

These analyses are included to encourage responsible interpretation of detector performance rather than relying solely on aggregate metrics such as accuracy or F1 score.

---

# References

See the project proposal, milestone reports, and course materials for the full methodology, literature review, benchmark rationale, and experimental design.

---

# Notes

* Large human-review datasets are not uploaded to GitHub.
* Earlier notebooks are preserved to document the evolution of the project.
* Model `.pkl` artifacts are included because later framework stages load pretrained classifiers directly.
* The updated detector uses a more diversified AI review dataset designed to reduce repetitive generation patterns.
* Even with diversified data, synthetic reviews may still contain residual stylistic artifacts that make classification easier than real-world mixed-origin text.

