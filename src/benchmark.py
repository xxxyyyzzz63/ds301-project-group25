from __future__ import annotations

import time
from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.pipeline import ReviewDetectionPipeline
from src.baseline_detector import BaselineDetector
from src.final_detector import FinalReviewDetector


class LLMBenchmark:
    """
    Benchmarks different approaches for AI review detection.

    Approaches tested:
    1. Baseline: simple few-shot LLM prompt
    2. Classifier-only: stylometric ML detector only
    3. Full pipeline (final fusion): uses fusion output directly
    4. Full pipeline (resolved): maps Uncertain back to classifier label

    Important:
    The full pipeline is run only once per review, and both "Final Fusion"
    and "Resolved" are derived from the same cached pipeline result.
    """

    def __init__(self) -> None:
        self.baseline_detector = BaselineDetector()
        self.classifier_only = FinalReviewDetector()
        self.full_pipeline = ReviewDetectionPipeline()

    @staticmethod
    def _compute_metrics(predictions: List[str], true_labels: List[str]) -> Dict[str, float]:
        y_true = [1 if label == "AI" else 0 for label in true_labels]
        y_pred = [1 if pred == "AI" else 0 for pred in predictions]

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _run_baseline(self, reviews: List[str]) -> Dict[str, Any]:
        predictions: List[str] = []
        total_time = 0.0

        for review in reviews:
            start_time = time.time()
            result = self.baseline_detector.detect(review)
            pred = "AI" if result.prediction.lower() == "ai" else "Human"
            predictions.append(pred)
            total_time += time.time() - start_time

        avg_time = total_time / len(reviews) if reviews else 0.0

        return {
            "approach": "Baseline (Few-Shot Prompt)",
            "predictions": predictions,
            "avg_time_per_review": avg_time,
            "total_time": total_time,
            "uncertain_count": 0,
        }

    def _run_classifier_only(self, reviews: List[str]) -> Dict[str, Any]:
        predictions: List[str] = []
        total_time = 0.0

        for review in reviews:
            start_time = time.time()
            result = self.classifier_only.detect(review)
            pred = result["predicted_label"]
            predictions.append(pred)
            total_time += time.time() - start_time

        avg_time = total_time / len(reviews) if reviews else 0.0

        return {
            "approach": "Classifier-Only (No LLM Fusion)",
            "predictions": predictions,
            "avg_time_per_review": avg_time,
            "total_time": total_time,
            "uncertain_count": 0,
        }

    def _run_full_pipeline_once(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Run the full pipeline once per review and derive both final-fusion and resolved outputs
        from the same cached pipeline result.
        """
        final_predictions: List[str] = []
        resolved_predictions: List[str] = []
        total_time = 0.0
        uncertain_count = 0

        for review in reviews:
            start_time = time.time()
            result = self.full_pipeline.run(review)
            total_time += time.time() - start_time

            fusion_pred = result["fusion_output"]["final_predicted_label"]
            classifier_pred = result["classifier_output"]["predicted_label"]

            final_predictions.append(fusion_pred)

            if fusion_pred == "Uncertain":
                uncertain_count += 1
                resolved_predictions.append(classifier_pred)
            else:
                resolved_predictions.append(fusion_pred)

        avg_time = total_time / len(reviews) if reviews else 0.0

        final_result = {
            "approach": "Full Pipeline (Final Fusion)",
            "predictions": final_predictions,
            "avg_time_per_review": avg_time,
            "total_time": total_time,
            "uncertain_count": uncertain_count,
        }

        resolved_result = {
            "approach": "Full Pipeline (Resolved)",
            "predictions": resolved_predictions,
            "avg_time_per_review": avg_time,
            "total_time": total_time,
            "uncertain_count": uncertain_count,
        }

        return {
            "final": final_result,
            "resolved": resolved_result,
        }

    def run_full_benchmark(
        self,
        test_reviews: List[str],
        test_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Run full benchmark across all approaches.
        """
        print("\n" + "=" * 100)
        print("LLM BENCHMARKING SUITE")
        print("=" * 100)
        print(f"\nTest set: {len(test_reviews)} reviews")
        print(
            f"Distribution: "
            f"{sum(1 for l in test_labels if l == 'AI')} AI, "
            f"{sum(1 for l in test_labels if l == 'Human')} Human\n"
        )

        print("Benchmarking: Baseline (Few-Shot Prompt)")
        print(f"Processing {len(test_reviews)} reviews...")
        baseline = self._run_baseline(test_reviews)

        print("\nBenchmarking: Classifier-Only (No LLM Fusion)")
        print(f"Processing {len(test_reviews)} reviews...")
        classifier_only = self._run_classifier_only(test_reviews)

        print("\nBenchmarking: Full Pipeline (Final Fusion + Resolved from same runs)")
        print(f"Processing {len(test_reviews)} reviews...")
        full_pipeline = self._run_full_pipeline_once(test_reviews)
        final_fusion = full_pipeline["final"]
        resolved = full_pipeline["resolved"]

        raw_results = [baseline, classifier_only, final_fusion, resolved]
        results = []

        for r in raw_results:
            metrics = self._compute_metrics(r["predictions"], test_labels)
            results.append({
                "approach": r["approach"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "avg_time_per_review": r["avg_time_per_review"],
                "total_time": r["total_time"],
                "uncertain_count": r["uncertain_count"],
                "predictions": r["predictions"],
            })

        winner = max(results, key=lambda x: x["f1"])["approach"] if results else None

        return {
            "test_size": len(test_reviews),
            "test_reviews": test_reviews,
            "true_labels": test_labels,
            "results": results,
            "winner": winner,
        }

    @staticmethod
    def pretty_print_benchmark(benchmark_results: Dict[str, Any]) -> None:
        """
        Pretty-print benchmark results.
        """
        print("\n" + "=" * 120)
        print("BENCHMARK RESULTS")
        print("=" * 120)

        results = benchmark_results["results"]

        print(
            f"\n{'Approach':<36} "
            f"{'Accuracy':<12} "
            f"{'Precision':<12} "
            f"{'Recall':<12} "
            f"{'F1':<12} "
            f"{'Time (s)':<12} "
            f"{'Uncertain':<12}"
        )
        print("-" * 120)

        for r in results:
            print(
                f"{r['approach']:<36} "
                f"{r['accuracy']:<12.4f} "
                f"{r['precision']:<12.4f} "
                f"{r['recall']:<12.4f} "
                f"{r['f1']:<12.4f} "
                f"{r['avg_time_per_review']:<12.4f} "
                f"{r['uncertain_count']:<12}"
            )

        print("\n" + "=" * 120)
        print(f"BEST APPROACH ON THIS TEST SET: {benchmark_results['winner']}")
        print("=" * 120)

        baseline = next(r for r in results if r["approach"] == "Baseline (Few-Shot Prompt)")
        classifier_only = next(r for r in results if r["approach"] == "Classifier-Only (No LLM Fusion)")
        final_fusion = next(r for r in results if r["approach"] == "Full Pipeline (Final Fusion)")
        resolved = next(r for r in results if r["approach"] == "Full Pipeline (Resolved)")

        print("\nKEY INSIGHTS:")

        if baseline["f1"] > 0:
            diff_final_vs_baseline = ((final_fusion["f1"] - baseline["f1"]) / baseline["f1"]) * 100
            diff_resolved_vs_baseline = ((resolved["f1"] - baseline["f1"]) / baseline["f1"]) * 100
            print(f"F1 difference: Final fusion vs baseline = {diff_final_vs_baseline:+.1f}%")
            print(f"F1 difference: Resolved pipeline vs baseline = {diff_resolved_vs_baseline:+.1f}%")
        else:
            print("Baseline F1 was 0.0, so percentage differences are undefined.")

        print(f"Classifier-only avg time: {classifier_only['avg_time_per_review']:.2f}s per review")
        print(f"Final fusion avg time: {final_fusion['avg_time_per_review']:.2f}s per review")
        print(f"Resolved pipeline avg time: {resolved['avg_time_per_review']:.2f}s per review")
        print(f"Final fusion uncertain outputs: {final_fusion['uncertain_count']}")
        print(f"Resolved pipeline uncertain outputs: {resolved['uncertain_count']}")

        if benchmark_results["test_size"] < 20:
            print(
                "\nCAUTION: This benchmark uses a very small test set. "
                "Treat these results as a smoke test, not a reliable performance comparison."
            )

            print("\nPer-example labels for inspection:")
            print("-" * 120)
            true_labels = benchmark_results["true_labels"]
            reviews = benchmark_results["test_reviews"]
            pred_lookup = {r["approach"]: r["predictions"] for r in results}

            for i, (review, true_label) in enumerate(zip(reviews, true_labels), start=1):
                review_short = review if len(review) <= 70 else review[:67] + "..."
                print(f"\nExample {i}")
                print(f"Review:   {review_short}")
                print(f"True:     {true_label}")
                for approach_name, preds in pred_lookup.items():
                    print(f"{approach_name:<36} {preds[i - 1]}")

    @staticmethod
    def benchmark_from_dataframe(
        df,
        review_col: str = "Review",
        label_col: str = "label"
    ) -> Dict[str, List[str]]:
        """
        Helper to extract review texts and labels from a dataframe.
        """
        reviews = df[review_col].astype(str).tolist()

        labels = []
        for value in df[label_col]:
            if isinstance(value, str):
                v = value.strip().lower()
                labels.append("AI" if v == "ai" else "Human")
            else:
                labels.append("AI" if int(value) == 1 else "Human")

        return {
            "reviews": reviews,
            "labels": labels,
        }


if __name__ == "__main__":
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY")
        raise SystemExit(1)

    test_reviews = [
        "stayed here last week. wifi terrible. breakfast meh but location good",
        "This establishment exceeded all expectations with impeccable service.",
        "Room 305 was clean. Staff helpful. Would stay again.",
        "The hotel provided exceptional accommodations with world-class amenities.",
    ]

    test_labels = ["Human", "AI", "Human", "AI"]

    benchmark = LLMBenchmark()
    results = benchmark.run_full_benchmark(test_reviews, test_labels)
    benchmark.pretty_print_benchmark(results)

    print("\nThis benchmark currently demonstrates:")
    print("- basic end-to-end comparison across approaches")
    print("- timing differences across systems")
    print("- difference between final fusion and resolved pipeline behavior")
    print("- whether the full pipeline produces uncertain cases")
    print("\nFor a meaningful benchmark, run this on a larger labeled review set.\n")
