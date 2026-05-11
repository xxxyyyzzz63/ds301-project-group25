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

The final system is intentionally more cautious than the classifier-only detector. The classifier remains strong as a raw detector, while the fusion layer adds interpretability and uncertainty-awareness.

# Recommended Entry Point
The easiest single entry point is:

```bash
python -m scripts.run_full_demo

This script:
- asks for a pasted hotel review
- runs the full live pipeline on that review
- prints Stage 1 linguistic analysis, Stage 2 classifier evidence, and Stage 3 fusion output
- summarizes saved pipeline test results
- summarizes disagreement analysis
- summarizes attribution analysis
- summarizes component validation
- summarizes the prompt framework
- summarizes the latest saved benchmark
- ends with the final decision for the input review: Human, AI, or Uncertain

## System Architecture 

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

## Final Project Status
The repository now includes:

- a few-shot prompting-only LLM baseline
- a stylometry-based classifier with calibrated probability outputs
- an LLM linguistic analyzer for structured semantic analysis
- an LLM fusion adjudicator that combines linguistic and classifier evidence
- a disagreement analyzer for uncertain or conflicting cases
- an attribution analyzer for identifying which linguistic signals mattered most
- a component validator for consistency, grounding, schema, and saved-output checks
- a prompt framework documenting the design of each LLM component
- benchmark and demo scripts for end-to-end evaluation and grading
 
The diversified-data update made the classification task substantially harder for logistic regression, while the final selected random forest model still performed extremely strongly. However, additional robustness analysis revealed that some human-written reviews can still be misclassified as strongly AI-generated, suggesting that residual stylometric shortcuts or synthetic dataset artifacts may remain.

---

# Repository Structure

```text
ds301-project-group25/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── tripadvisor_hotel_reviews.csv
│   ├── ai_generated_tripadvisor_reviews_gemma3_4b.csv
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
├── outputs/
│   ├── pipeline_test_results.csv
│   ├── pipeline_test_summary.json
│   ├── disagreement_summary.json
│   ├── attribution_analysis_results.json
│   ├── component_validation_summary.json
│   ├── prompt_framework.json
│   ├── benchmark_results_*.json
│   ├── benchmark_results_*.csv
│   └── full_demo_runs.json
│
├── scripts/
│   ├── run_demo.py
│   ├── run_disagreement_analysis.py
│   ├── run_attribution_analysis.py
│   ├── run_component_validation.py
│   ├── run_prompt_framework.py
│   ├── run_benchmark.py
│   └── run_full_demo.py
│
└── src/
    ├── __init__.py
    ├── baseline_detector.py
    ├── stylometry_features.py
    ├── final_detector.py
    ├── llm_linguistic_analyzer.py
    ├── fusion_adjudicator.py
    ├── pipeline.py
    ├── disagreement_analyzer.py
    ├── attribution_analyzer.py
    ├── component_validator.py
    ├── prompt_framework.py
    └── benchmark.py

```

---

# Setup Instructions

1. Clone the repository.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the dataset files are available in the data/ folder:

- tripadvisor_hotel_reviews.csv
- ai_generated_tripadvisor_reviews_gemma3_4b.csv
- ai_generated_tripadvisor_reviews_openai_diverse.csv

4. Set your OpenAI API key as an environment variable:

export OPENAI_API_KEY="your_key_here"

5. Run the full demo:

python -m scripts.run_full_demo

---

## Main Pipeline

The final live system is a three-stage pipeline:

## Stage 1: LLM Linguistic Analyzer
This module reads a review and outputs six structured linguistic dimensions:
- tone
- specificity
- personal experience markers
- templated language
- human messiness
- narrative flow
It also returns evidence spans and an overall linguistic assessment.

## Stage 2: Stylometric Classifier
This module computes the stylometric feature representation and returns:
- predicted label
- calibrated AI probability
- AI-likeness score
- uncertainty band
- top stylometric features
- classifier explanation

## Stage 3: LLM Fusion Adjudicator
This module combines Stage 1 and Stage 2 outputs and returns:
- agreement status
- final predicted label
- final uncertainty band
- final explanation

## Analysis and Validation Layer

## Disagreement Analyzer
Studies cases where classifier evidence and linguistic evidence conflict or where the final fused decision is uncertain.

## Attribution Analyzer
Identifies which linguistic signals contributed most in uncertain or conflicting cases, including supporting signals, conflicting signals, and key evidence spans.

## Component Validator
Checks:
- repeated-run consistency
- evidence grounding
- schema compliance
- saved pipeline-output consistency

## Prompt Framework
Documents the role, design logic, and prompting strategy for each LLM component so the LLM design is explicit and systematic.

## Benchmark
Compares:
Baseline (Few-Shot Prompt)
Classifier-Only (No LLM Fusion)
Full Pipeline (Final Fusion)
Full Pipeline (Resolved)

## Scripts
# Main demo
- scripts/run_full_demo.py: full grading entry point combining a live review analysis with summaries of saved outputs

# Other runnable scripts
- scripts/run_demo.py: simple live pipeline demo for one pasted review
- scripts/run_disagreement_analysis.py: builds and prints disagreement analysis summary
- scripts/run_attribution_analysis.py: builds and prints attribution-analysis results
- scripts/run_component_validation.py: runs the validator and saves validation summary
- scripts/run_prompt_framework.py: exports prompt-framework documentation
- scripts/run_benchmark.py: runs the benchmark on a sampled subset of AI and human reviews

# Source Code

# Core prediction modules
- src/baseline_detector.py: few-shot prompting-only baseline
- src/stylometry_features.py: stylometric feature extraction utilities
- src/final_detector.py: calibrated classifier interface
- src/llm_linguistic_analyzer.py: Stage 1 semantic analysis
- src/fusion_adjudicator.py: Stage 3 evidence fusion
- src/pipeline.py: end-to-end orchestration of the live pipeline

# Analysis modules
- src/disagreement_analyzer.py: disagreement and uncertainty analysis
- src/attribution_analyzer.py: linguistic attribution and counterfactual analysis
- src/benchmark.py: multi-approach benchmark suite

# Validation and documentation modules
- src/component_validator.py: LLM reliability and output consistency checks
- src/prompt_framework.py: structured documentation of prompt design

# Notebooks
The notebooks preserve the project’s earlier milestone workflow and intermediate experiments.
- notebooks/ai_review_generation_and_eda.ipynb: generates AI reviews and performs exploratory analysis
- notebooks/AI_Review_Detector_Week1_3_Complete.ipynb: Weeks 1 to 3 implementation notebook
- notebooks/data_preparation.ipynb: data cleaning and preprocessing
- notebooks/evaluate_baseline.ipynb: prompting-only baseline evaluation
- notebooks/week4_week5_updated_with_diverse_data.ipynb: updated Week 4 to 5 notebook using the diversified AI data
- notebooks/detector_sanity_check.ipynb: end-to-end detector loading and testing
- notebooks/subgroup_analysis_by_length.ipynb: subgroup analysis by review length
- notebooks/dataset_audit_and_shortcut_analysis.ipynb: dataset audit and leakage/shortcut analysis
- notebooks/casing_ood_experiment.ipynb: out-of-distribution casing experiment
- notebooks/final_demo.ipynb: end-to-end notebook demo on curated examples

# Model Artifacts
The repository keeps both earlier and final artifacts so the project evolution remains visible.
Important files include:
- diverse_week5_artifacts.pkl: main final detector artifact
- updated_week4_week5_outputs.pkl: updated evaluation outputs
- subgroup_analysis_outputs.pkl: saved subgroup analysis tables
- dataset_audit_outputs.pkl: saved audit outputs
- casing_ood_experiment_outputs.pkl: saved casing experiment outputs

# Notes
- Large CSV files may be omitted from GitHub depending on storage limits.
- The final system depends on both saved model artifacts and live OpenAI API access for the LLM stages.
- Earlier notebooks reflect the original milestone workflow.
- The updated final pipeline is intentionally more interpretable and cautious than the classifier-only detector.
- Results should still be interpreted carefully because synthetic AI reviews may retain residual cues that differ from real-world mixed-origin text.
---

# References

See the milestone reports and proposal for the full methodology, literature review, design rationale, and evaluation logic.

---

