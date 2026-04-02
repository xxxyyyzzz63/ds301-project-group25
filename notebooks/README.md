# AI-Generated Hotel Review Detector

**Course**: DS-UA 301 - Advanced Topics in Data Science (NYU Spring 2025)  
**Team**: Wendy, Wency, Yujia

## Overview

This project investigates whether a stylometry-based, feature-driven detector can distinguish AI-generated hotel reviews from human-written reviews more reliably and transparently than a single-pass prompting-only LLM baseline. Our system is designed as a multi-step LangChain-style pipeline: a review is converted into stylometric features, scored by a trained classifier, calibrated into a 0–100 AI-likeness score, and then explained using evidence grounded in the extracted features.

## Research Question

Can a stylometry-based, feature-driven detector with structured outputs produce more reliable AI-vs-human review detection and clearer evidence-based explanations than a single-pass prompting-only LLM baseline?

## Repository Structure

```
ds301-project-group25/
├── README.md                          # Main project overview and setup instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── models/                            # Saved model artifacts
│   ├── baseline_results.pkl           # Week 2 baseline evaluation results
│   ├── lr_classifier.pkl              # Logistic regression classifier
│   ├── lr_temp_scaler.pkl             # Logistic regression temperature scaler
│   ├── rf_classifier.pkl              # Random forest classifier
│   ├── rf_temp_scaler.pkl             # Random forest temperature scaler
│   └── week3_artifacts.pkl            # Week 3 saved outputs and metadata
│
├── notebooks/                         # Development notebooks
│   ├── AI_Review_Detector_Week1_3_Complete.ipynb   # Main Weeks 1–3 implementation notebook
│   ├── ai_review_generation_and_eda.ipynb          # AI review generation + proposal EDA notebook
│   ├── data_preparation.ipynb                      # Dataset cleaning and preprocessing
│   ├── evaluate_baseline.ipynb                     # Baseline evaluation notebook
│   └── README.md                                   # Notes about notebook contents
│
├── src/                               # Python source code
│   ├── baseline_detector.py           # Week 2 prompting-only baseline logic
│   └── stylometry_features.py         # Stylometric feature extraction code
│
└── Data files (not uploaded to GitHub):
    ├── tripadvisor_hotel_reviews.csv
    └── ai_generated_tripadvisor_reviews_gemma3_4b.csv

```
