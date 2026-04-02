# AI-Generated Hotel Review Detector

**Course**: DS-UA 301 - Advanced Topics in Data Science (NYU Spring 2025)  
**Team**: Wendy, Wency, Yujia

## Overview

Detecting AI-generated hotel reviews using stylometric features and a multi-step LangChain pipeline.

## Research Question

Can a stylometry-based, feature-driven detector with structured outputs produce more reliable AI-vs-human review detection and clearer evidence-based explanations than a single-pass prompting-only LLM baseline?

## Project Structure
```
DS301_AI_Review_Detector/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── Project.ipynb                # Main project notebook
│
└── Data files (NOT on GitHub - too large):
    ├── tripadvisor_hotel_reviews.csv              (~15MB - human reviews)
    └── ai_generated_tripadvisor_reviews_gemma3_4b.csv  (~7MB - AI reviews)
```

## Setup Instructions

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download datasets separately (see proposal for sources)
4. Set OpenAI API key in environment

## Weekly Progress

- ✓ Week 1: Project setup, data collection (10,000 AI reviews generated)
- □ Week 2: Baseline LLM detector implementation
- □ Week 3: Classifier training and calibration  
- □ Week 4: Full LangChain pipeline
- □ Week 5: Evaluation and analysis

## References

See `Project_Proposal.pdf` for full methodology, literature review, and implementation details.

---

**Note**: Large data files (CSV) are excluded from this repository. Team members should download datasets separately.
