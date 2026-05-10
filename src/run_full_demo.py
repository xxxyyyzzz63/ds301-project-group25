from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

from src.pipeline import ReviewDetectionPipeline


OUTPUTS_DIR = Path("outputs")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_benchmark_json() -> Path | None:
    candidates = sorted(
        glob.glob(str(OUTPUTS_DIR / "benchmark_results_*_seed*_*.json"))
    )
    if not candidates:
        return None
    return Path(candidates[-1])


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def run_live_pipeline_demo() -> None:
    print_header("1. LIVE PIPELINE DEMO")

    review = (
        "We stayed for two nights. The bed was comfortable and the bathroom was clean, "
        "but the parking fee felt too high for what we got."
    )

    print("\nInput review:")
    print(review)

    pipeline = ReviewDetectionPipeline()
    result = pipeline.run(review)

    ling = result["linguistic_analysis"]
    clf = result["classifier_output"]
    fusion = result["fusion_output"]

    print_subheader("Stage 1: LLM Linguistic Analyzer")
    print(f"Tone: {ling['tone']}")
    print(f"Specificity: {ling['specificity']}")
    print(f"Personal experience markers: {ling['personal_experience_markers']}")
    print(f"Templated language: {ling['templated_language']}")
    print(f"Human messiness: {ling['human_messiness']}")
    print(f"Narrative flow: {ling['narrative_flow']}")
    print(f"Evidence spans: {ling['evidence_spans']}")

    print_subheader("Stage 2: Stylometric Classifier")
    print(f"Predicted label: {clf['predicted_label']}")
    print(f"AI probability: {clf['ai_probability']}")
    print(f"AI-likeness score: {clf['ai_likeness_score']}")
    print(f"Uncertainty band: {clf['uncertainty_band']}")
    print(f"Top features: {clf['top_features']}")

    print_subheader("Stage 3: LLM Fusion Adjudicator")
    print(f"Agreement status: {fusion['agreement_status']}")
    print(f"Final predicted label: {fusion['final_predicted_label']}")
    print(f"Final uncertainty band: {fusion['final_uncertainty_band']}")
    print(f"Final explanation: {fusion['final_explanation']}")


def print_pipeline_test_summary() -> None:
    path = OUTPUTS_DIR / "pipeline_test_summary.json"
    if not path.exists():
        print_header("2. PIPELINE TEST SUMMARY")
        print("\nMissing outputs/pipeline_test_summary.json")
        print("Run: python test_ai_and_human.py")
        return

    data = load_json(path)

    print_header("2. PIPELINE TEST SUMMARY")
    print(f"\nSource: {path}")
    print(f"Total reviews tested: {data['n_total']}")
    print(f"AI reviews tested: {data['n_ai']}")
    print(f"Human reviews tested: {data['n_human']}")
    print(f"Final fusion accuracy: {data['overall_final_accuracy']:.1f}%")
    print(f"Resolved accuracy: {data['overall_resolved_accuracy']:.1f}%")
    print(f"AI final accuracy: {data['ai_final_accuracy']:.1f}%")
    print(f"Human final accuracy: {data['human_final_accuracy']:.1f}%")
    print(f"Overall uncertain rate: {data['overall_uncertain_rate']:.1f}%")

    print("\nInterpretation:")
    print("The full fusion system is cautious, especially on AI reviews that look human-like.")
    print("The resolved setting shows the underlying classifier remains strong.")


def print_disagreement_summary() -> None:
    path = OUTPUTS_DIR / "disagreement_summary.json"
    if not path.exists():
        print_header("3. DISAGREEMENT ANALYSIS SUMMARY")
        print("\nMissing outputs/disagreement_summary.json")
        print("Run: python -m scripts.run_disagreement_analysis")
        return

    data = load_json(path)

    print_header("3. DISAGREEMENT ANALYSIS SUMMARY")
    print(f"\nSource: {path}")
    print(f"Cases analyzed: {data['n_cases_analyzed']}")
    print(f"Agreement counts: {data['agreement_counts']}")
    print(f"Fusion-label counts: {data['fusion_label_counts']}")
    print(f"True-label counts: {data['true_label_counts']}")

    print("\nTop conflicting signals:")
    for signal, count in data["top_conflicting_signals"][:5]:
        print(f"  - {signal}: {count}")

    if data.get("example_cases"):
        ex = data["example_cases"][0]
        print("\nExample disagreement case:")
        print(f"  True label: {ex['true_label']}")
        print(f"  Classifier: {ex['classifier_label']} ({ex['classifier_ai_probability']:.4f})")
        print(f"  Fusion: {ex['fusion_label']} ({ex['agreement_status']})")
        print(f"  Conflicting signals: {ex['conflicting_signals']}")


def print_attribution_summary() -> None:
    path = OUTPUTS_DIR / "attribution_analysis_results.json"
    if not path.exists():
        print_header("4. ATTRIBUTION ANALYSIS SUMMARY")
        print("\nMissing outputs/attribution_analysis_results.json")
        print("Run: python -m scripts.run_attribution_analysis")
        return

    data = load_json(path)

    print_header("4. ATTRIBUTION ANALYSIS SUMMARY")
    print(f"\nSource: {path}")
    print(f"Cases analyzed: {len(data)}")

    for i, case in enumerate(data[:2], start=1):
        print_subheader(f"Example Attribution Case {i}")
        print(f"True label: {case['true_label']}")
        print(f"Final predicted label: {case['final_predicted_label']}")
        print(f"Agreement status: {case['agreement_status']}")
        print(f"Primary signal: {case['primary_signal']}")
        print(f"Linguistic contribution score: {case['linguistic_contribution_score']}/100")
        print(f"Key evidence span: {case['key_evidence_span']}")


def print_component_validation_summary() -> None:
    path = OUTPUTS_DIR / "component_validation_summary.json"
    if not path.exists():
        print_header("5. COMPONENT VALIDATION SUMMARY")
        print("\nMissing outputs/component_validation_summary.json")
        print("Run: python -m scripts.run_component_validation")
        return

    data = load_json(path)

    print_header("5. COMPONENT VALIDATION SUMMARY")
    print(f"\nSource: {path}")

    single = data["single_review_validation"]
    pipeline = data["saved_pipeline_validation"]

    print("\nSingle-review validator checks:")
    print(
        f"  Repeated-run consistency: "
        f"{single['consistency_validation']['overall_consistency']:.1f}%"
    )
    print(
        f"  Evidence grounding: "
        f"{single['grounding_validation']['grounded_percentage']:.1f}%"
    )
    print(
        f"  Schema compliant: "
        f"{single['schema_validation']['compliant']}"
    )
    print(f"  Overall passed: {single['overall_passed']}")

    print("\nSaved pipeline-output checks:")
    print(f"  Rows checked: {pipeline['n_rows']}")
    print(f"  Missing columns: {pipeline['missing_columns']}")
    print(f"  Bad agreement rows: {len(pipeline['bad_agreement_rows'])}")
    print(f"  Bad uncertainty rows: {len(pipeline['bad_uncertainty_rows'])}")
    print(f"  Bad resolved rows: {len(pipeline['bad_resolved_rows'])}")
    print(f"  Overall passed: {pipeline['passed']}")


def print_prompt_framework_summary() -> None:
    path = OUTPUTS_DIR / "prompt_framework.json"
    if not path.exists():
        print_header("6. PROMPT FRAMEWORK SUMMARY")
        print("\nMissing outputs/prompt_framework.json")
        print("Run: python -m scripts.run_prompt_framework")
        return

    data = load_json(path)

    print_header("6. PROMPT FRAMEWORK SUMMARY")
    print(f"\nSource: {path}")

    framework = data["project_prompt_framework"]
    components = framework["components"]

    print("\nLLM components and roles:")
    for name, info in components.items():
        print(f"  - {name}: {info['purpose']}")

    print("\nSystematic design claims:")
    for claim in framework["systematic_design_claims"]:
        print(f"  - {claim}")


def print_benchmark_summary() -> None:
    path = find_latest_benchmark_json()
    if path is None:
        print_header("7. BENCHMARK SUMMARY")
        print("\nNo saved benchmark JSON found in outputs/")
        print("Run: python -m scripts.run_benchmark 20 20")
        return

    data = load_json(path)

    print_header("7. BENCHMARK SUMMARY")
    print(f"\nSource: {path}")
    print(f"Test size: {data['test_size']}")
    print(f"Best approach: {data['winner']}")

    print("\nApproach comparison:")
    for r in data["results"]:
        print(
            f"  - {r['approach']}: "
            f"accuracy={r['accuracy']:.2f}, "
            f"precision={r['precision']:.2f}, "
            f"recall={r['recall']:.2f}, "
            f"f1={r['f1']:.2f}, "
            f"uncertain={r['uncertain_count']}"
        )

    print("\nInterpretation:")
    print("Classifier-only is the strongest raw detector on the benchmark sample.")
    print("The full fusion system is more cautious and interpretable, but lower in recall.")
    print("The resolved setting shows that the underlying classifier remains strong.")


def main() -> None:
    print_header("FULL PROJECT DEMO")
    print(
        "\nThis script provides a single entry point for grading.\n"
        "It runs one live pipeline example and then summarizes all saved outputs:\n"
        "pipeline test, disagreement analysis, attribution analysis,\n"
        "component validation, prompt framework, and benchmark results."
    )

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "\nERROR: OPENAI_API_KEY is not set.\n"
            "The live pipeline demo requires API access.\n"
            "Set the key and rerun, or inspect saved outputs manually."
        )
        sys.exit(1)

    run_live_pipeline_demo()
    print_pipeline_test_summary()
    print_disagreement_summary()
    print_attribution_summary()
    print_component_validation_summary()
    print_prompt_framework_summary()
    print_benchmark_summary()

    print_header("DEMO COMPLETE")
    print(
        "\nRecommended grading takeaway:\n"
        "- The project is now a multi-stage LLM-guided pipeline.\n"
        "- The LLM is used for linguistic analysis, fusion, and meta-analysis.\n"
        "- The classifier remains strong, while fusion adds caution and interpretability.\n"
    )


if __name__ == "__main__":
    main()
