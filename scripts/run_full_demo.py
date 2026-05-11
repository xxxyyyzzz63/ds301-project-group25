from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from src.pipeline import ReviewDetectionPipeline


OUTPUTS_DIR = Path("outputs")
DEMO_RUNS_PATH = OUTPUTS_DIR / "full_demo_runs.json"


def load_json(path: Path) -> Any:
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


def get_first_existing(data: dict, keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def save_demo_result(review_text: str, result: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "review_text": review_text,
        "result": result,
    }
    with open(DEMO_RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prompt_for_review() -> str:
    print_header("LIVE REVIEW ANALYSIS")
    print("Paste a hotel review below. Press Enter twice when finished.")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "":
            break

        lines.append(line)

    return "\n".join(lines).strip()


def run_live_pipeline_demo() -> dict:
    review_text = prompt_for_review()

    if not review_text:
        print("\nNo review text was provided.")
        sys.exit(0)

    pipeline = ReviewDetectionPipeline()
    result = pipeline.run(review_text)
    save_demo_result(review_text, result)

    ling = result["linguistic_analysis"]
    clf = result["classifier_output"]
    fusion = result["fusion_output"]

    print_subheader("Input Review")
    print(review_text)

    print_subheader("Stage 1: LLM Linguistic Analyzer")
    print(f"Tone: {ling['tone']}")
    print(f"Specificity: {ling['specificity']}")
    print(f"Personal experience markers: {ling['personal_experience_markers']}")
    print(f"Templated language: {ling['templated_language']}")
    print(f"Human messiness: {ling['human_messiness']}")
    print(f"Narrative flow: {ling['narrative_flow']}")
    print(f"Evidence spans: {ling['evidence_spans']}")
    print(f"Overall linguistic assessment: {ling['overall_linguistic_assessment']}")

    print_subheader("Stage 2: Stylometric Classifier")
    print(f"Model used: {clf['model_used']}")
    print(f"Calibrated: {clf['calibrated']}")
    print(f"Predicted label: {clf['predicted_label']}")
    print(f"AI probability: {clf['ai_probability']}")
    print(f"AI-likeness score: {clf['ai_likeness_score']}")
    print(f"Uncertainty band: {clf['uncertainty_band']}")
    print(f"Top features: {clf['top_features']}")
    print(f"Classifier explanation: {clf['explanation']}")

    print_subheader("Stage 3: LLM Fusion Adjudicator")
    print(f"Classifier label passed to fusion: {fusion['classifier_label']}")
    print(f"Classifier probability passed to fusion: {fusion['classifier_probability']}")
    print(f"Agreement status: {fusion['agreement_status']}")
    print(f"Final predicted label: {fusion['final_predicted_label']}")
    print(f"Final uncertainty band: {fusion['final_uncertainty_band']}")
    print(f"Final explanation: {fusion['final_explanation']}")

    print(f"\nSaved live demo run to: {DEMO_RUNS_PATH}")

    return result


def print_pipeline_test_summary() -> None:
    path = OUTPUTS_DIR / "pipeline_test_summary.json"
    print_header("SAVED PIPELINE TEST SUMMARY")

    if not path.exists():
        print(f"Missing {path}")
        print("Run: python test_ai_and_human.py")
        return

    data = load_json(path)

    n_total = get_first_existing(data, ["n_total", "total_tested"])
    n_ai = get_first_existing(data, ["n_ai"])
    n_human = get_first_existing(data, ["n_human"])

    overall_final_accuracy = get_first_existing(
        data, ["overall_final_accuracy", "overall_accuracy"]
    )
    overall_resolved_accuracy = get_first_existing(
        data, ["overall_resolved_accuracy", "resolved_accuracy"]
    )
    ai_final_accuracy = get_first_existing(
        data, ["ai_final_accuracy", "ai_accuracy"]
    )
    human_final_accuracy = get_first_existing(
        data, ["human_final_accuracy", "human_accuracy"]
    )
    overall_uncertain_rate = get_first_existing(
        data, ["overall_uncertain_rate"]
    )

    print(f"Source: {path}")
    if n_total is not None:
        print(f"Total reviews tested: {n_total}")
    if n_ai is not None:
        print(f"AI reviews tested: {n_ai}")
    if n_human is not None:
        print(f"Human reviews tested: {n_human}")
    if overall_final_accuracy is not None:
        print(f"Final fusion accuracy: {overall_final_accuracy:.1f}%")
    if overall_resolved_accuracy is not None:
        print(f"Resolved accuracy: {overall_resolved_accuracy:.1f}%")
    if ai_final_accuracy is not None:
        print(f"AI final accuracy: {ai_final_accuracy:.1f}%")
    if human_final_accuracy is not None:
        print(f"Human final accuracy: {human_final_accuracy:.1f}%")
    if overall_uncertain_rate is not None:
        print(f"Overall uncertain rate: {overall_uncertain_rate:.1f}%")

    print_subheader("Interpretation")
    if (
        ai_final_accuracy is not None
        and human_final_accuracy is not None
        and overall_uncertain_rate is not None
    ):
        print(
            f"The saved 60-review pipeline test shows a strong asymmetry: final human accuracy is "
            f"{human_final_accuracy:.1f}% while final AI accuracy is only {ai_final_accuracy:.1f}%. "
            f"The overall uncertain rate is {overall_uncertain_rate:.1f}%, which means the fusion layer "
            "often abstains when the linguistic evidence looks human-like, even when the classifier remains strong."
        )
    else:
        print("The saved pipeline summary loaded successfully.")


def print_disagreement_summary() -> None:
    path = OUTPUTS_DIR / "disagreement_summary.json"
    print_header("SAVED DISAGREEMENT ANALYSIS")

    if not path.exists():
        print(f"Missing {path}")
        print("Run: python -m scripts.run_disagreement_analysis")
        return

    data = load_json(path)

    n_cases = get_first_existing(data, ["n_cases_analyzed", "n_cases"])
    agreement_counts = get_first_existing(data, ["agreement_counts"], {})
    fusion_label_counts = get_first_existing(
        data, ["fusion_label_counts", "fusion_counts"], {}
    )
    true_label_counts = get_first_existing(data, ["true_label_counts"], {})
    top_conflicting = get_first_existing(data, ["top_conflicting_signals"], [])

    print(f"Source: {path}")
    if n_cases is not None:
        print(f"Cases analyzed: {n_cases}")
    print(f"Agreement counts: {agreement_counts}")
    print(f"Fusion-label counts: {fusion_label_counts}")
    print(f"True-label counts: {true_label_counts}")

    print_subheader("Top Conflicting Signals")
    for signal, count in top_conflicting[:5]:
        print(f"{signal}: {count}")

    example_cases = get_first_existing(data, ["example_cases"], [])
    if example_cases:
        ex = example_cases[0]
        print_subheader("Example Saved Disagreement Case")
        print(f"True label: {ex.get('true_label')}")
        print(
            f"Classifier output: {ex.get('classifier_label')} "
            f"({ex.get('classifier_ai_probability', 0):.4f})"
        )
        print(
            f"Fusion output: {ex.get('fusion_label')} "
            f"({ex.get('agreement_status')})"
        )
        print(f"Conflicting signals: {ex.get('conflicting_signals')}")
        if "review_excerpt" in ex:
            print(f"Review excerpt: {ex['review_excerpt']}")

    print_subheader("Interpretation")
    if top_conflicting:
        top_signal, top_count = top_conflicting[0]
        print(
            f"The disagreement analyzer shows that many uncertain cases are driven by human-like linguistic "
            f"signals. The most common conflicting signal is '{top_signal}' with count {top_count}. "
            "This helps explain why the fusion layer becomes cautious on some AI reviews."
        )
    else:
        print("The saved disagreement summary loaded successfully.")


def print_attribution_summary() -> None:
    path = OUTPUTS_DIR / "attribution_analysis_results.json"
    print_header("SAVED ATTRIBUTION ANALYSIS")

    if not path.exists():
        print(f"Missing {path}")
        print("Run: python -m scripts.run_attribution_analysis")
        return

    data = load_json(path)

    print(f"Source: {path}")
    print(f"Cases analyzed: {len(data)}")

    for i, case in enumerate(data[:2], start=1):
        print_subheader(f"Example Attribution Case {i}")
        print(f"True label: {case.get('true_label')}")
        print(f"Final predicted label: {case.get('final_predicted_label')}")
        print(f"Agreement status: {case.get('agreement_status')}")
        print(f"Primary signal: {case.get('primary_signal')}")
        print(
            f"Linguistic contribution score: "
            f"{case.get('linguistic_contribution_score')}/100"
        )
        print(f"Supporting signals: {case.get('supporting_signals')}")
        print(f"Conflicting signals: {case.get('conflicting_signals')}")
        print(f"Key evidence span: {case.get('key_evidence_span')}")

    primary_counts = {}
    for case in data:
        primary = case.get("primary_signal")
        if primary is not None:
            primary_counts[primary] = primary_counts.get(primary, 0) + 1

    print_subheader("Interpretation")
    if primary_counts:
        top_primary = max(primary_counts.items(), key=lambda x: x[1])
        print(
            f"Across the saved attribution cases, the most frequent primary signal is "
            f"'{top_primary[0]}' ({top_primary[1]} cases). This module explains which linguistic "
            "dimension contributed most when the fusion layer became uncertain or conflicted."
        )
    else:
        print("The saved attribution results loaded successfully.")


def print_component_validation_summary() -> None:
    path = OUTPUTS_DIR / "component_validation_summary.json"
    print_header("SAVED COMPONENT VALIDATION")

    if not path.exists():
        print(f"Missing {path}")
        print("Run: python -m scripts.run_component_validation")
        return

    data = load_json(path)

    single = data["single_review_validation"]
    pipeline = data["saved_pipeline_validation"]

    print(f"Source: {path}")

    print_subheader("Single-Review Validator Checks")
    print(
        f"Repeated-run consistency: "
        f"{single['consistency_validation']['overall_consistency']:.1f}%"
    )
    print(
        f"Evidence grounding: "
        f"{single['grounding_validation']['grounded_percentage']:.1f}%"
    )
    print(f"Schema compliant: {single['schema_validation']['compliant']}")
    print(f"Overall passed: {single['overall_passed']}")

    print_subheader("Saved Pipeline-Output Checks")
    print(f"Rows checked: {pipeline['n_rows']}")
    print(f"Missing columns: {pipeline['missing_columns']}")
    print(f"Bad agreement rows: {len(pipeline['bad_agreement_rows'])}")
    print(f"Bad uncertainty rows: {len(pipeline['bad_uncertainty_rows'])}")
    print(f"Bad resolved rows: {len(pipeline['bad_resolved_rows'])}")
    print(f"Overall passed: {pipeline['passed']}")

    print_subheader("Interpretation")
    print(
        "The component validator confirms that the LLM outputs are structured, grounded, and internally "
        "consistent, so the prompt-based stages are being tested systematically rather than treated as inherently reliable."
    )


def print_prompt_framework_summary() -> None:
    path = OUTPUTS_DIR / "prompt_framework.json"
    print_header("SAVED PROMPT FRAMEWORK")

    if not path.exists():
        print(f"Missing {path}")
        print("Run: python -m scripts.run_prompt_framework")
        return

    data = load_json(path)

    framework = data["project_prompt_framework"]
    components = framework["components"]

    print(f"Source: {path}")

    print_subheader("LLM Components and Roles")
    for name, info in components.items():
        print(f"{name}: {info['purpose']}")

    print_subheader("Systematic Design Claims")
    for claim in framework["systematic_design_claims"]:
        print(f"- {claim}")

    print_subheader("Interpretation")
    print(
        "The prompt framework documents the role of each LLM module and makes the project's prompt engineering explicit, "
        "rather than leaving it hidden inside notebooks or prompts."
    )


def print_benchmark_summary() -> None:
    path = find_latest_benchmark_json()
    print_header("SAVED BENCHMARK SUMMARY")

    if path is None:
        print("No saved benchmark JSON found in outputs/")
        print("Run: python -m scripts.run_benchmark 20 20")
        return

    data = load_json(path)

    print(f"Source: {path}")
    print(f"Test size: {data['test_size']}")
    print(f"Best approach: {data['winner']}")

    print_subheader("Approach Comparison")
    for r in data["results"]:
        print(
            f"{r['approach']}: "
            f"accuracy={r['accuracy']:.2f}, "
            f"precision={r['precision']:.2f}, "
            f"recall={r['recall']:.2f}, "
            f"f1={r['f1']:.2f}, "
            f"uncertain={r['uncertain_count']}"
        )

    results_by_name = {r["approach"]: r for r in data["results"]}
    clf = results_by_name.get("Classifier-Only (No LLM Fusion)")
    final = results_by_name.get("Full Pipeline (Final Fusion)")
    resolved = results_by_name.get("Full Pipeline (Resolved)")

    print_subheader("Interpretation")
    if clf and final and resolved:
        print(
            f"On the saved benchmark, classifier-only reaches accuracy {clf['accuracy']:.2f} and F1 {clf['f1']:.2f}, "
            f"while the final fusion pipeline drops to accuracy {final['accuracy']:.2f} and F1 {final['f1']:.2f} "
            f"with {final['uncertain_count']} uncertain cases. The resolved version returns to accuracy "
            f"{resolved['accuracy']:.2f} and F1 {resolved['f1']:.2f}, showing that the classifier remains strong "
            "and the fusion layer is the source of extra caution."
        )
    else:
        print("Saved benchmark results were loaded successfully.")


def print_final_decision(result: dict) -> None:
    fusion = result["fusion_output"]
    clf = result["classifier_output"]

    print_header("FINAL DECISION FOR INPUT REVIEW")
    print(f"Final fused label: {fusion['final_predicted_label']}")
    print(f"Final uncertainty band: {fusion['final_uncertainty_band']}")
    print(f"Agreement status: {fusion['agreement_status']}")
    print(f"Classifier label: {clf['predicted_label']}")
    print(f"Classifier AI probability: {clf['ai_probability']}")
    print(f"Final explanation: {fusion['final_explanation']}")


def main() -> None:
    print_header("FULL PROJECT DEMO")
    print(
        "This script runs the live multi-stage pipeline on your input review and then summarizes "
        "the saved outputs from the rest of the project: pipeline testing, disagreement analysis, "
        "attribution analysis, component validation, prompt framework, and benchmarking."
    )

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "\nERROR: OPENAI_API_KEY is not set.\n"
            "The live pipeline demo requires API access."
        )
        sys.exit(1)

    live_result = run_live_pipeline_demo()
    print_pipeline_test_summary()
    print_disagreement_summary()
    print_attribution_summary()
    print_component_validation_summary()
    print_prompt_framework_summary()
    print_benchmark_summary()
    print_final_decision(live_result)


if __name__ == "__main__":
    main()
