from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Dict, Any, List

import pandas as pd

from src.pipeline import ReviewDetectionPipeline


def _ensure_outputs_dir() -> None:
    os.makedirs("outputs", exist_ok=True)


def _safe_get_top_features(classifier_output: Dict[str, Any]) -> Dict[str, Any]:
    top_features = classifier_output.get("top_features", {})
    if isinstance(top_features, dict):
        return top_features
    return {}


def _flatten_result(
    review_text: str,
    true_label: str,
    pipeline_result: Dict[str, Any],
) -> Dict[str, Any]:
    ling = pipeline_result["linguistic_analysis"]
    clf = pipeline_result["classifier_output"]
    fusion = pipeline_result["fusion_output"]
    top_features = _safe_get_top_features(clf)

    final_label = fusion.get("final_predicted_label")
    classifier_label = clf.get("predicted_label")
    resolved_label = classifier_label if final_label == "Uncertain" else final_label

    row = {
        "review_text": review_text,
        "true_label": true_label,

        "ling_tone": ling.get("tone"),
        "ling_specificity": ling.get("specificity"),
        "ling_personal_experience_markers": ling.get("personal_experience_markers"),
        "ling_templated_language": ling.get("templated_language"),
        "ling_human_messiness": ling.get("human_messiness"),
        "ling_narrative_flow": ling.get("narrative_flow"),
        "ling_evidence_spans": json.dumps(ling.get("evidence_spans", []), ensure_ascii=False),
        "ling_overall_assessment": ling.get("overall_linguistic_assessment"),

        "classifier_model_used": clf.get("model_used"),
        "classifier_calibrated": clf.get("calibrated"),
        "classifier_predicted_label": classifier_label,
        "classifier_ai_probability": clf.get("ai_probability"),
        "classifier_ai_likeness_score": clf.get("ai_likeness_score"),
        "classifier_uncertainty_band": clf.get("uncertainty_band"),
        "classifier_explanation": clf.get("explanation"),
        "classifier_top_features": json.dumps(top_features, ensure_ascii=False),

        "fusion_classifier_label": fusion.get("classifier_label"),
        "fusion_classifier_probability": fusion.get("classifier_probability"),
        "fusion_agreement_status": fusion.get("agreement_status"),
        "fusion_final_predicted_label": final_label,
        "fusion_final_uncertainty_band": fusion.get("final_uncertainty_band"),
        "fusion_final_explanation": fusion.get("final_explanation"),

        "resolved_label": resolved_label,
        "final_correct": final_label == true_label,
        "resolved_correct": resolved_label == true_label,
    }

    return row


def test_both_datasets(
    n_ai: int = 30,
    n_human: int = 30,
    random_state: int = 42,
    save_csv_path: str = "outputs/pipeline_test_results.csv",
    save_json_path: str = "outputs/pipeline_test_summary.json",
) -> Dict[str, Any]:
    _ensure_outputs_dir()

    print("\n" + "=" * 100)
    print("COMPREHENSIVE PIPELINE TEST - AI vs HUMAN")
    print("=" * 100)

    print("\nLoading datasets...")
    ai_df = pd.read_csv("data/ai_generated_tripadvisor_reviews_openai_diverse.csv")
    human_df = pd.read_csv("data/tripadvisor_hotel_reviews.csv")

    print(f"AI reviews available: {len(ai_df)}")
    print(f"Human reviews available: {len(human_df)}")

    if "Review" not in ai_df.columns:
        raise ValueError("AI dataset must contain a 'Review' column.")
    if "Review" not in human_df.columns:
        raise ValueError("Human dataset must contain a 'Review' column.")

    if "label" not in ai_df.columns:
        ai_df["label"] = "AI"
    if "label" not in human_df.columns:
        human_df["label"] = "Human"

    print("\nInitializing pipeline...")
    pipeline = ReviewDetectionPipeline()
    print("Pipeline ready!")

    n_ai = min(n_ai, len(ai_df))
    n_human = min(n_human, len(human_df))

    ai_sample = ai_df.sample(n=n_ai, random_state=random_state).reset_index(drop=True)
    human_sample = human_df.sample(n=n_human, random_state=random_state).reset_index(drop=True)

    print(f"\nTesting {n_ai} AI and {n_human} Human reviews...")
    print("This may take several minutes depending on sample size.\n")

    all_rows: List[Dict[str, Any]] = []
    ai_results = []
    human_results = []

    print("=" * 100)
    print("PART 1: TESTING AI REVIEWS")
    print("=" * 100)

    for _, row in ai_sample.iterrows():
        try:
            review_text = str(row["Review"])
            result = pipeline.run(review_text)

            final = result["fusion_output"]["final_predicted_label"]
            agreement = result["fusion_output"]["agreement_status"]
            resolved = result["classifier_output"]["predicted_label"] if final == "Uncertain" else final

            ling = result["linguistic_analysis"]
            human_like = (
                ling["tone"] in ["casual", "mixed"]
                and ling["specificity"] in ["concrete", "mixed"]
                and ling["personal_experience_markers"] in ["strong", "moderate"]
                and ling["templated_language"] == "low"
            )

            ai_results.append({
                "final": final,
                "resolved": resolved,
                "agreement": agreement,
                "tone": ling["tone"],
                "specificity": ling["specificity"],
                "templated": ling["templated_language"],
                "human_like": human_like,
                "final_correct": final == "AI",
                "resolved_correct": resolved == "AI",
            })

            all_rows.append(
                _flatten_result(
                    review_text=review_text,
                    true_label="AI",
                    pipeline_result=result,
                )
            )

            match = "OK" if final == "AI" else "WRONG" if final == "Human" else "UNCERTAIN"
            flag = "HUMAN-LIKE" if human_like else ""
            print(f"  [{len(ai_results):2d}/{n_ai}] {final:10s} ({agreement:8s}) {match:10s} {flag}")

        except Exception as e:
            print(f"  [{len(ai_results)+1:2d}/{n_ai}] ERROR: {str(e)[:70]}")

    print("\n" + "=" * 100)
    print("PART 2: TESTING HUMAN REVIEWS")
    print("=" * 100)

    for _, row in human_sample.iterrows():
        try:
            review_text = str(row["Review"])
            result = pipeline.run(review_text)

            final = result["fusion_output"]["final_predicted_label"]
            agreement = result["fusion_output"]["agreement_status"]
            resolved = result["classifier_output"]["predicted_label"] if final == "Uncertain" else final

            ling = result["linguistic_analysis"]
            ai_like = (
                ling["tone"] == "polished"
                and ling["specificity"] == "generic"
                and ling["templated_language"] in ["high", "moderate"]
            )

            human_results.append({
                "final": final,
                "resolved": resolved,
                "agreement": agreement,
                "tone": ling["tone"],
                "ai_like": ai_like,
                "final_correct": final == "Human",
                "resolved_correct": resolved == "Human",
            })

            all_rows.append(
                _flatten_result(
                    review_text=review_text,
                    true_label="Human",
                    pipeline_result=result,
                )
            )

            match = "OK" if final == "Human" else "WRONG" if final == "AI" else "UNCERTAIN"
            flag = "AI-LIKE" if ai_like else ""
            print(f"  [{len(human_results):2d}/{n_human}] {final:10s} ({agreement:8s}) {match:10s} {flag}")

        except Exception as e:
            print(f"  [{len(human_results)+1:2d}/{n_human}] ERROR: {str(e)[:70]}")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(save_csv_path, index=False)
    print(f"\nSaved row-level results to: {save_csv_path}")

    print("\n" + "=" * 100)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 100)

    print("\nAI REVIEWS ANALYSIS")
    print("-" * 100)

    ai_final_correct = sum(1 for r in ai_results if r["final_correct"])
    ai_resolved_correct = sum(1 for r in ai_results if r["resolved_correct"])
    ai_uncertain = sum(1 for r in ai_results if r["final"] == "Uncertain")
    ai_wrong_human = sum(1 for r in ai_results if r["final"] == "Human")
    ai_human_like = sum(1 for r in ai_results if r["human_like"])

    print(f"\n  Final predictions:")
    print(f"    Correct (AI):      {ai_final_correct}/{len(ai_results)} = {ai_final_correct/len(ai_results)*100:.1f}%")
    print(f"    Uncertain:         {ai_uncertain}/{len(ai_results)} = {ai_uncertain/len(ai_results)*100:.1f}%")
    print(f"    Wrong (Human):     {ai_wrong_human}/{len(ai_results)} = {ai_wrong_human/len(ai_results)*100:.1f}%")
    print(f"\n  Resolved accuracy (Uncertain -> classifier label):")
    print(f"    Correct after resolution: {ai_resolved_correct}/{len(ai_results)} = {ai_resolved_correct/len(ai_results)*100:.1f}%")

    print(f"\n  Linguistic Characteristics:")
    print(f"    Human-like signals: {ai_human_like}/{len(ai_results)} = {ai_human_like/len(ai_results)*100:.1f}%")

    ai_agreement = Counter(r["agreement"] for r in ai_results)
    print(f"\n  Agreement Status:")
    print(f"    Agree:    {ai_agreement['agree']}")
    print(f"    Mixed:    {ai_agreement['mixed']}")
    print(f"    Disagree: {ai_agreement['disagree']}")

    ai_tones = Counter(r["tone"] for r in ai_results)
    print(f"\n  Tone Distribution:")
    for tone, count in ai_tones.most_common():
        print(f"    {tone:10s}: {count} ({count/len(ai_results)*100:.1f}%)")

    print("\nHUMAN REVIEWS ANALYSIS")
    print("-" * 100)

    human_final_correct = sum(1 for r in human_results if r["final_correct"])
    human_resolved_correct = sum(1 for r in human_results if r["resolved_correct"])
    human_uncertain = sum(1 for r in human_results if r["final"] == "Uncertain")
    human_wrong_ai = sum(1 for r in human_results if r["final"] == "AI")
    human_ai_like = sum(1 for r in human_results if r["ai_like"])

    print(f"\n  Final predictions:")
    print(f"    Correct (Human):   {human_final_correct}/{len(human_results)} = {human_final_correct/len(human_results)*100:.1f}%")
    print(f"    Uncertain:         {human_uncertain}/{len(human_results)} = {human_uncertain/len(human_results)*100:.1f}%")
    print(f"    Wrong (AI):        {human_wrong_ai}/{len(human_results)} = {human_wrong_ai/len(human_results)*100:.1f}%")
    print(f"\n  Resolved accuracy (Uncertain -> classifier label):")
    print(f"    Correct after resolution: {human_resolved_correct}/{len(human_results)} = {human_resolved_correct/len(human_results)*100:.1f}%")

    print(f"\n  Linguistic Characteristics:")
    print(f"    AI-like signals:   {human_ai_like}/{len(human_results)} = {human_ai_like/len(human_results)*100:.1f}%")

    human_agreement = Counter(r["agreement"] for r in human_results)
    print(f"\n  Agreement Status:")
    print(f"    Agree:    {human_agreement['agree']}")
    print(f"    Mixed:    {human_agreement['mixed']}")
    print(f"    Disagree: {human_agreement['disagree']}")

    print("\nOVERALL STATISTICS")
    print("-" * 100)

    total_final_correct = ai_final_correct + human_final_correct
    total_resolved_correct = ai_resolved_correct + human_resolved_correct
    total_tested = len(ai_results) + len(human_results)
    total_uncertain = ai_uncertain + human_uncertain

    print(f"\n  Final fusion accuracy:   {total_final_correct}/{total_tested} = {total_final_correct/total_tested*100:.1f}%")
    print(f"  Resolved accuracy:       {total_resolved_correct}/{total_tested} = {total_resolved_correct/total_tested*100:.1f}%")
    print(f"  AI final accuracy:       {ai_final_correct}/{len(ai_results)} = {ai_final_correct/len(ai_results)*100:.1f}%")
    print(f"  AI resolved accuracy:    {ai_resolved_correct}/{len(ai_results)} = {ai_resolved_correct/len(ai_results)*100:.1f}%")
    print(f"  Human final accuracy:    {human_final_correct}/{len(human_results)} = {human_final_correct/len(human_results)*100:.1f}%")
    print(f"  Human resolved accuracy: {human_resolved_correct}/{len(human_results)} = {human_resolved_correct/len(human_results)*100:.1f}%")
    print(f"  Total Uncertain:         {total_uncertain}/{total_tested} = {total_uncertain/total_tested*100:.1f}%")

    print("\nKEY INSIGHTS")
    print("-" * 100)

    if ai_human_like / len(ai_results) > 0.7:
        print("\n  EXCELLENT AI QUALITY: >70% of AI reviews have human-like linguistic signals")
        print("  Your diverse dataset demonstrates sophisticated AI generation.")

    if ai_uncertain / len(ai_results) > 0.5:
        print("\n  CAUTIOUS FUSION: >50% of AI reviews flagged as uncertain")
        print("  System is surfacing conflicts between statistical and linguistic evidence.")

    if human_final_correct / len(human_results) > 0.6:
        print("\n  STRONG HUMAN DETECTION: >60% accuracy on human reviews")
        print("  Linguistic analysis is successfully identifying genuine human writing.")

    disagree_count = ai_agreement["disagree"] + human_agreement["disagree"]
    if disagree_count > 10:
        print(f"\n  VALUABLE EDGE CASES: {disagree_count} disagreement cases detected")
        print("  These are ideal inputs for a disagreement analyzer.")

    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)

    summary = {
        "n_ai": len(ai_results),
        "n_human": len(human_results),
        "total_tested": total_tested,

        "ai_final_accuracy": ai_final_correct / len(ai_results) * 100 if ai_results else 0.0,
        "ai_resolved_accuracy": ai_resolved_correct / len(ai_results) * 100 if ai_results else 0.0,
        "human_final_accuracy": human_final_correct / len(human_results) * 100 if human_results else 0.0,
        "human_resolved_accuracy": human_resolved_correct / len(human_results) * 100 if human_results else 0.0,
        "overall_final_accuracy": total_final_correct / total_tested * 100 if total_tested else 0.0,
        "overall_resolved_accuracy": total_resolved_correct / total_tested * 100 if total_tested else 0.0,

        "ai_uncertain_rate": ai_uncertain / len(ai_results) * 100 if ai_results else 0.0,
        "human_uncertain_rate": human_uncertain / len(human_results) * 100 if human_results else 0.0,
        "overall_uncertain_rate": total_uncertain / total_tested * 100 if total_tested else 0.0,

        "ai_human_like_rate": ai_human_like / len(ai_results) * 100 if ai_results else 0.0,
        "human_ai_like_rate": human_ai_like / len(human_results) * 100 if human_results else 0.0,

        "ai_agreement_counts": dict(ai_agreement),
        "human_agreement_counts": dict(human_agreement),
        "save_csv_path": save_csv_path,
    }

    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to: {save_json_path}")

    print("\nSummary for report:")
    print(f"  - Tested {total_tested} reviews ({len(ai_results)} AI, {len(human_results)} Human)")
    print(f"  - Final fusion accuracy: {summary['overall_final_accuracy']:.1f}%")
    print(f"  - Resolved accuracy: {summary['overall_resolved_accuracy']:.1f}%")
    print(f"  - AI reviews with human-like signals: {summary['ai_human_like_rate']:.1f}%")
    print(f"  - Human reviews correctly identified (final): {summary['human_final_accuracy']:.1f}%")
    print(f"  - Uncertain classifications: {total_uncertain} ({summary['overall_uncertain_rate']:.1f}%)")

    return summary


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    N_AI = 30
    N_HUMAN = 30
    RANDOM_STATE = 42

    try:
        results = test_both_datasets(
            n_ai=N_AI,
            n_human=N_HUMAN,
            random_state=RANDOM_STATE,
            save_csv_path="outputs/pipeline_test_results.csv",
            save_json_path="outputs/pipeline_test_summary.json",
        )
        print("\nTest completed successfully!")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
