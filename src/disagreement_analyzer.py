from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List

import pandas as pd

from src.pipeline import ReviewDetectionPipeline


class DisagreementAnalyzer:
    """
    Analyzes disagreements between classifier and fusion outputs.

    This component is meant to show when the linguistic-analysis stage
    changes, softens, or complicates the classifier-only decision.
    """

    def __init__(self) -> None:
        self.pipeline = ReviewDetectionPipeline()

    def analyze_single_review(self, review_text: str) -> Dict[str, Any]:
        result = self.pipeline.run(review_text)

        clf_label = result["classifier_output"]["predicted_label"]
        clf_prob = float(result["classifier_output"]["ai_probability"])

        fusion_label = result["fusion_output"]["final_predicted_label"]
        agreement_status = result["fusion_output"]["agreement_status"]

        decision_changed = clf_label != fusion_label and fusion_label != "Uncertain"
        flagged_uncertain = fusion_label == "Uncertain"

        ling = result["linguistic_analysis"]
        conflicting_signals = self._identify_conflicts(
            clf_label=clf_label,
            clf_prob=clf_prob,
            linguistic_analysis=ling,
        )

        return {
            "review_text": review_text,
            "true_label": None,
            "classifier_label": clf_label,
            "classifier_probability": clf_prob,
            "fusion_label": fusion_label,
            "agreement_status": agreement_status,
            "decision_changed": decision_changed,
            "flagged_uncertain": flagged_uncertain,
            "linguistic_analysis": ling,
            "conflicting_signals": conflicting_signals,
            "classifier_output": result["classifier_output"],
            "fusion_output": result["fusion_output"],
            "full_result": result,
        }

    @staticmethod
    def _identify_conflicts(
        clf_label: str,
        clf_prob: float,
        linguistic_analysis: Dict[str, Any]
    ) -> List[str]:
        conflicts: List[str] = []

        if clf_label == "AI" and clf_prob >= 0.70:
            if linguistic_analysis.get("specificity") == "concrete":
                conflicts.append("specificity = concrete")
            if linguistic_analysis.get("personal_experience_markers") == "strong":
                conflicts.append("personal_experience_markers = strong")
            if linguistic_analysis.get("human_messiness") in {"high", "moderate"}:
                conflicts.append("human_messiness = moderate/high")
            if linguistic_analysis.get("templated_language") == "low":
                conflicts.append("templated_language = low")
            if linguistic_analysis.get("tone") in {"casual", "mixed"}:
                conflicts.append("tone = casual/mixed")

        elif clf_label == "Human" and clf_prob <= 0.30:
            if linguistic_analysis.get("specificity") == "generic":
                conflicts.append("specificity = generic")
            if linguistic_analysis.get("tone") == "polished":
                conflicts.append("tone = polished")
            if linguistic_analysis.get("templated_language") in {"high", "moderate"}:
                conflicts.append("templated_language = moderate/high")
            if linguistic_analysis.get("narrative_flow") == "formulaic":
                conflicts.append("narrative_flow = formulaic")

        return conflicts

    @staticmethod
    def _row_to_case(row: pd.Series) -> Dict[str, Any]:
        evidence_spans = []
        top_features = {}

        if pd.notna(row.get("ling_evidence_spans")):
            try:
                evidence_spans = json.loads(row["ling_evidence_spans"])
            except Exception:
                evidence_spans = []

        if pd.notna(row.get("classifier_top_features")):
            try:
                top_features = json.loads(row["classifier_top_features"])
            except Exception:
                top_features = {}

        linguistic_analysis = {
            "tone": row.get("ling_tone"),
            "specificity": row.get("ling_specificity"),
            "personal_experience_markers": row.get("ling_personal_experience_markers"),
            "templated_language": row.get("ling_templated_language"),
            "human_messiness": row.get("ling_human_messiness"),
            "narrative_flow": row.get("ling_narrative_flow"),
            "evidence_spans": evidence_spans,
            "overall_linguistic_assessment": row.get("ling_overall_assessment"),
        }

        classifier_output = {
            "predicted_label": row.get("classifier_predicted_label"),
            "ai_probability": float(row.get("classifier_ai_probability")),
            "ai_likeness_score": row.get("classifier_ai_likeness_score"),
            "uncertainty_band": row.get("classifier_uncertainty_band"),
            "top_features": top_features,
            "explanation": row.get("classifier_explanation"),
        }

        fusion_output = {
            "classifier_label": row.get("fusion_classifier_label"),
            "classifier_probability": row.get("fusion_classifier_probability"),
            "agreement_status": row.get("fusion_agreement_status"),
            "final_predicted_label": row.get("fusion_final_predicted_label"),
            "final_uncertainty_band": row.get("fusion_final_uncertainty_band"),
            "final_explanation": row.get("fusion_final_explanation"),
        }

        clf_label = classifier_output["predicted_label"]
        clf_prob = float(classifier_output["ai_probability"])
        fusion_label = fusion_output["final_predicted_label"]
        agreement_status = fusion_output["agreement_status"]

        return {
            "review_text": row.get("review_text"),
            "true_label": row.get("true_label"),
            "classifier_label": clf_label,
            "classifier_probability": clf_prob,
            "fusion_label": fusion_label,
            "agreement_status": agreement_status,
            "decision_changed": clf_label != fusion_label and fusion_label != "Uncertain",
            "flagged_uncertain": fusion_label == "Uncertain",
            "linguistic_analysis": linguistic_analysis,
            "conflicting_signals": DisagreementAnalyzer._identify_conflicts(
                clf_label=clf_label,
                clf_prob=clf_prob,
                linguistic_analysis=linguistic_analysis,
            ),
            "classifier_output": classifier_output,
            "fusion_output": fusion_output,
        }

    def analyze_results_dataframe(
        self,
        df: pd.DataFrame,
        only_flagged_cases: bool = True
    ) -> Dict[str, Any]:
        required_cols = {
            "review_text",
            "true_label",
            "classifier_predicted_label",
            "classifier_ai_probability",
            "fusion_agreement_status",
            "fusion_final_predicted_label",
            "ling_tone",
            "ling_specificity",
            "ling_personal_experience_markers",
            "ling_templated_language",
            "ling_human_messiness",
            "ling_narrative_flow",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in results CSV: {sorted(missing)}")

        cases = [self._row_to_case(row) for _, row in df.iterrows()]

        if only_flagged_cases:
            filtered_cases = [
                c for c in cases
                if c["agreement_status"] == "disagree" or c["fusion_label"] == "Uncertain"
            ]
        else:
            filtered_cases = cases

        agreement_counts = Counter(c["agreement_status"] for c in filtered_cases)
        fusion_counts = Counter(c["fusion_label"] for c in filtered_cases)
        true_label_counts = Counter(c["true_label"] for c in filtered_cases)

        conflict_counter = Counter()
        conflict_counter_disagree = Counter()
        conflict_counter_mixed = Counter()

        for c in filtered_cases:
            for signal in c["conflicting_signals"]:
                conflict_counter[signal] += 1
                if c["agreement_status"] == "disagree":
                    conflict_counter_disagree[signal] += 1
                elif c["agreement_status"] == "mixed":
                    conflict_counter_mixed[signal] += 1

        by_true_label = {}
        for label in sorted(set(c["true_label"] for c in filtered_cases if c["true_label"] is not None)):
            subset = [c for c in filtered_cases if c["true_label"] == label]
            by_true_label[label] = {
                "n_cases": len(subset),
                "agreement_counts": dict(Counter(c["agreement_status"] for c in subset)),
                "fusion_counts": dict(Counter(c["fusion_label"] for c in subset)),
            }

        return {
            "n_total_rows": len(df),
            "n_analyzed_cases": len(filtered_cases),
            "agreement_counts": dict(agreement_counts),
            "fusion_counts": dict(fusion_counts),
            "true_label_counts": dict(true_label_counts),
            "top_conflicting_signals": conflict_counter.most_common(10),
            "top_conflicting_signals_disagree": conflict_counter_disagree.most_common(10),
            "top_conflicting_signals_mixed": conflict_counter_mixed.most_common(10),
            "by_true_label": by_true_label,
            "cases": filtered_cases,
        }

    def analyze_results_csv(
        self,
        csv_path: str,
        only_flagged_cases: bool = True
    ) -> Dict[str, Any]:
        df = pd.read_csv(csv_path)
        return self.analyze_results_dataframe(df, only_flagged_cases=only_flagged_cases)

    @staticmethod
    def pretty_print_summary(summary: Dict[str, Any], n_examples: int = 5) -> None:
        print("\n" + "=" * 100)
        print("DISAGREEMENT ANALYSIS SUMMARY")
        print("=" * 100)

        print(f"\nTotal rows in results file: {summary['n_total_rows']}")
        print(f"Cases analyzed: {summary['n_analyzed_cases']}")

        print("\nAgreement counts:")
        for k, v in summary["agreement_counts"].items():
            print(f"  {k}: {v}")

        print("\nFusion-label counts:")
        for k, v in summary["fusion_counts"].items():
            print(f"  {k}: {v}")

        print("\nTrue-label counts among analyzed cases:")
        for k, v in summary["true_label_counts"].items():
            print(f"  {k}: {v}")

        print("\nTop conflicting signals overall:")
        for signal, count in summary["top_conflicting_signals"]:
            print(f"  {signal:<40} {count}")

        print("\nTop conflicting signals in disagree cases:")
        for signal, count in summary["top_conflicting_signals_disagree"]:
            print(f"  {signal:<40} {count}")

        print("\nTop conflicting signals in mixed cases:")
        for signal, count in summary["top_conflicting_signals_mixed"]:
            print(f"  {signal:<40} {count}")

        print("\nBy true label:")
        for label, stats in summary["by_true_label"].items():
            print(f"  {label}:")
            print(f"    n_cases: {stats['n_cases']}")
            print(f"    agreement_counts: {stats['agreement_counts']}")
            print(f"    fusion_counts: {stats['fusion_counts']}")

        print("\nExample cases:")
        for i, case in enumerate(summary["cases"][:n_examples], start=1):
            print("\n" + "-" * 100)
            print(f"Example {i}")
            print(f"True label: {case['true_label']}")
            print(f"Classifier: {case['classifier_label']} ({case['classifier_probability']:.4f})")
            print(f"Fusion: {case['fusion_label']} ({case['agreement_status']})")
            print(f"Conflicting signals: {case['conflicting_signals']}")
            print(f"Review: {str(case['review_text'])[:180]}...")

        print("\n" + "=" * 100)