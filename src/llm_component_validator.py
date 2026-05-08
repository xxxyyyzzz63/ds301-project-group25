from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List

import pandas as pd

from src.llm_linguistic_analyzer import LLMLinguisticAnalyzer
from src.pipeline import ReviewDetectionPipeline


class LLMComponentValidator:
    """
    Systematic validator for LLM-based components in the project.

    Validates:
    - Stage 1 linguistic analyzer consistency
    - evidence grounding
    - schema compliance
    - saved pipeline output consistency from outputs/pipeline_test_results.csv
    """

    LINGUISTIC_DIMENSIONS = [
        "tone",
        "specificity",
        "personal_experience_markers",
        "templated_language",
        "human_messiness",
        "narrative_flow",
    ]

    EXPECTED_VALUES = {
        "tone": ["casual", "polished", "mixed"],
        "specificity": ["concrete", "generic", "mixed"],
        "personal_experience_markers": ["strong", "moderate", "weak"],
        "templated_language": ["high", "moderate", "low"],
        "human_messiness": ["high", "moderate", "low"],
        "narrative_flow": ["natural", "formulaic", "mixed"],
    }

    def __init__(self) -> None:
        self.linguistic_analyzer = LLMLinguisticAnalyzer()
        self.pipeline = ReviewDetectionPipeline()

    def validate_consistency(
        self,
        review_text: str,
        n_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Run the linguistic analyzer multiple times and check consistency.
        """
        print(f"\nRunning consistency test ({n_runs} runs)...")

        results = []
        for i in range(n_runs):
            result = self.linguistic_analyzer.analyze(review_text)
            results.append(result)
            print(f"  Run {i + 1}/{n_runs} complete")

        agreement_scores: Dict[str, Any] = {}
        for dim in self.LINGUISTIC_DIMENSIONS:
            values = [r[dim] for r in results]
            most_common_value, count = Counter(values).most_common(1)[0]
            agreement_pct = (count / n_runs) * 100
            agreement_scores[dim] = {
                "agreement_pct": agreement_pct,
                "most_common": most_common_value,
                "all_values": values,
            }

        avg_agreement = sum(
            s["agreement_pct"] for s in agreement_scores.values()
        ) / len(self.LINGUISTIC_DIMENSIONS)

        return {
            "n_runs": n_runs,
            "overall_consistency": avg_agreement,
            "dimension_agreement": agreement_scores,
            "interpretation": (
                "Excellent (>=90%)"
                if avg_agreement >= 90
                else "Good (70-90%)"
                if avg_agreement >= 70
                else "Poor (<70%)"
            ),
        }

    def validate_evidence_grounding(
        self,
        review_text: str,
        linguistic_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check whether evidence spans actually appear in the input review.
        """
        print("\nValidating evidence grounding...")

        evidence_spans = linguistic_analysis.get("evidence_spans", [])
        if not evidence_spans:
            return {
                "grounded": True,
                "warning": "No evidence spans provided",
                "missing_spans": [],
                "grounded_percentage": 100.0,
            }

        missing_spans = []
        for span in evidence_spans:
            if span.lower() not in review_text.lower():
                missing_spans.append(span)

        grounded_pct = (
            ((len(evidence_spans) - len(missing_spans)) / len(evidence_spans)) * 100
            if evidence_spans
            else 100.0
        )

        return {
            "grounded": len(missing_spans) == 0,
            "grounded_percentage": grounded_pct,
            "total_spans": len(evidence_spans),
            "grounded_spans": len(evidence_spans) - len(missing_spans),
            "missing_spans": missing_spans,
            "interpretation": (
                "All grounded" if len(missing_spans) == 0 else f"{len(missing_spans)} not found"
            ),
        }

    def validate_schema_compliance(
        self,
        linguistic_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate that the linguistic analyzer output follows the expected schema.
        """
        print("\nValidating schema compliance...")

        required_fields = self.LINGUISTIC_DIMENSIONS + [
            "evidence_spans",
            "overall_linguistic_assessment",
        ]

        missing_fields = [f for f in required_fields if f not in linguistic_analysis]

        invalid_values = {}
        for field, allowed in self.EXPECTED_VALUES.items():
            if field in linguistic_analysis:
                value = linguistic_analysis[field]
                if value not in allowed:
                    invalid_values[field] = {
                        "actual": value,
                        "expected": allowed,
                    }

        evidence_valid = isinstance(linguistic_analysis.get("evidence_spans", []), list)

        compliant = (
            len(missing_fields) == 0
            and len(invalid_values) == 0
            and evidence_valid
        )

        return {
            "compliant": compliant,
            "missing_fields": missing_fields,
            "invalid_values": invalid_values,
            "evidence_spans_is_list": evidence_valid,
            "interpretation": "Fully compliant" if compliant else "Validation failed",
        }

    def run_full_validation(
        self,
        review_text: str,
        n_consistency_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Run validation on the linguistic analyzer directly.
        """
        print("\n" + "=" * 100)
        print("LLM COMPONENT VALIDATION SUITE")
        print("=" * 100)
        print(f"\nReview: {review_text[:120]}...\n")

        linguistic_analysis = self.linguistic_analyzer.analyze(review_text)

        consistency_result = self.validate_consistency(review_text, n_consistency_runs)
        grounding_result = self.validate_evidence_grounding(review_text, linguistic_analysis)
        schema_result = self.validate_schema_compliance(linguistic_analysis)

        all_passed = (
            consistency_result["overall_consistency"] >= 70
            and grounding_result["grounded"]
            and schema_result["compliant"]
        )

        return {
            "review_text": review_text,
            "consistency_validation": consistency_result,
            "grounding_validation": grounding_result,
            "schema_validation": schema_result,
            "overall_passed": all_passed,
            "summary": (
                "ALL PASSED - LLM component is systematic"
                if all_passed
                else "SOME CHECKS FAILED - review component behavior"
            ),
        }

    def validate_saved_pipeline_outputs(
        self,
        csv_path: str = "outputs/pipeline_test_results.csv",
    ) -> Dict[str, Any]:
        """
        Validate consistency of saved pipeline outputs from test_ai_and_human.py.
        """
        print("\n" + "=" * 100)
        print("PIPELINE OUTPUT VALIDATION")
        print("=" * 100)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Run test_ai_and_human.py first."
            )

        df = pd.read_csv(csv_path)
        print(f"\nLoaded {len(df)} saved pipeline rows from {csv_path}")

        required_cols = [
            "review_text",
            "true_label",
            "classifier_predicted_label",
            "classifier_ai_probability",
            "fusion_agreement_status",
            "fusion_final_predicted_label",
            "fusion_final_uncertainty_band",
            "resolved_label",
            "ling_tone",
            "ling_specificity",
            "ling_personal_experience_markers",
            "ling_templated_language",
            "ling_human_messiness",
            "ling_narrative_flow",
        ]

        missing_cols = [c for c in required_cols if c not in df.columns]

        bad_agreement_rows = []
        bad_uncertainty_rows = []
        bad_resolved_rows = []

        for idx, row in df.iterrows():
            final_label = row["fusion_final_predicted_label"]
            agreement = row["fusion_agreement_status"]
            uncertainty_band = row["fusion_final_uncertainty_band"]
            classifier_label = row["classifier_predicted_label"]
            resolved_label = row["resolved_label"]

            if final_label == "Uncertain" and agreement not in {"mixed", "disagree"}:
                bad_agreement_rows.append(idx)

            if final_label == "AI" and uncertainty_band != "likely AI-generated":
                bad_uncertainty_rows.append(idx)
            elif final_label == "Human" and uncertainty_band != "likely human-written":
                bad_uncertainty_rows.append(idx)
            elif final_label == "Uncertain" and uncertainty_band != "uncertain":
                bad_uncertainty_rows.append(idx)

            expected_resolved = classifier_label if final_label == "Uncertain" else final_label
            if resolved_label != expected_resolved:
                bad_resolved_rows.append(idx)

        summary = {
            "n_rows": len(df),
            "missing_columns": missing_cols,
            "bad_agreement_rows": bad_agreement_rows,
            "bad_uncertainty_rows": bad_uncertainty_rows,
            "bad_resolved_rows": bad_resolved_rows,
            "agreement_counts": dict(Counter(df["fusion_agreement_status"])),
            "final_label_counts": dict(Counter(df["fusion_final_predicted_label"])),
            "passed": (
                len(missing_cols) == 0
                and len(bad_agreement_rows) == 0
                and len(bad_uncertainty_rows) == 0
                and len(bad_resolved_rows) == 0
            ),
        }

        return summary

    @staticmethod
    def pretty_print_validation(validation_result: Dict[str, Any]) -> None:
        print("\n" + "=" * 100)
        print("VALIDATION RESULTS")
        print("=" * 100)

        cons = validation_result["consistency_validation"]
        print(f"\n1. CONSISTENCY ({cons['n_runs']} runs)")
        print(f"   Overall: {cons['overall_consistency']:.1f}% agreement")
        print(f"   {cons['interpretation']}")

        ground = validation_result["grounding_validation"]
        print(f"\n2. EVIDENCE GROUNDING")
        print(f"   {ground['interpretation']}")

        schema = validation_result["schema_validation"]
        print(f"\n3. SCHEMA COMPLIANCE")
        print(f"   {schema['interpretation']}")

        print(f"\nOVERALL: {validation_result['summary']}")
        print("=" * 100 + "\n")

    @staticmethod
    def pretty_print_pipeline_validation(summary: Dict[str, Any]) -> None:
        print("\n" + "=" * 100)
        print("PIPELINE VALIDATION SUMMARY")
        print("=" * 100)

        print(f"\nRows checked: {summary['n_rows']}")
        print(f"Missing columns: {summary['missing_columns']}")
        print(f"Bad agreement rows: {len(summary['bad_agreement_rows'])}")
        print(f"Bad uncertainty rows: {len(summary['bad_uncertainty_rows'])}")
        print(f"Bad resolved rows: {len(summary['bad_resolved_rows'])}")

        print("\nAgreement counts:")
        for k, v in summary["agreement_counts"].items():
            print(f"  {k}: {v}")

        print("\nFinal label counts:")
        for k, v in summary["final_label_counts"].items():
            print(f"  {k}: {v}")

        print(f"\nOverall passed: {summary['passed']}")
        print("=" * 100 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING: LLM Component Validator")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set")
        raise SystemExit(1)

    validator = LLMComponentValidator()

    test_review = "stayed here last week. room was clean but breakfast meh."
    validation = validator.run_full_validation(test_review, n_consistency_runs=3)
    validator.pretty_print_validation(validation)

    if os.path.exists("outputs/pipeline_test_results.csv"):
        pipeline_summary = validator.validate_saved_pipeline_outputs(
            "outputs/pipeline_test_results.csv"
        )
        validator.pretty_print_pipeline_validation(pipeline_summary)

    print("=" * 80)
    print("Component Validator working correctly!")
    print("=" * 80 + "\n")
