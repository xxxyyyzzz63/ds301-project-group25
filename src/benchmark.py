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
    3. Full pipeline: multi-stage LLM-guided system
    """

    def __init__(self) -> None:
        self.baseline_detector = BaselineDetector()
        self.classifier_only = FinalReviewDetector()
        self.full_pipeline = ReviewDetectionPipeline()

    def benchmark_approach(
        self,
        approach_name: str,
        reviews: List[str],
        true_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Benchmark a single approach.
        """
        print(f"\nBenchmarking: {approach_name}")
        print(f"Processing {len(reviews)} reviews...")

        predictions = []
        total_time = 0.0
        uncertain_count = 0

        for review in reviews:
            start_time = time.time()

            if approach_name == "Baseline (Few-Shot Prompt)":
                result = self.baseline_detector.detect(review)
                pred = "AI" if result.prediction.lower() == "ai" else "Human"

            elif approach_name == "Classifier-Only (No LLM Fusion)":
                result = self.classifier_only.detect(review)
                pred = result["predicted_label"]

            elif approach_name == "Full Pipeline (Multi-Stage LLM)":
                result = self.full_pipeline.run(review)
                pred = result["fusion_output"]["final_predicted_label"]

                if pred == "Uncertain":
                    uncertain_count += 1
                    # For benchmark scoring, map Uncertain back to classifier decision
                    pred = result["classifier_output"]["predicted_label"]

            else:
                raise ValueError(f"Unknown approach: {approach_name}")

            predictions.append(pred)
            total_time += time.time() - start_time

        y_true = [1 if label == "AI" else 0 for label in true_labels]
        y_pred = [1 if pred == "AI" else 0 for pred in predictions]

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        avg_time = total_time / len(reviews) if reviews else 0.0

        return {
            "approach": approach_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_time_per_review": avg_time,
            "total_time": total_time,
            "uncertain_count": uncertain_count,
            "predictions": predictions,
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

        approaches = [
            "Baseline (Few-Shot Prompt)",
            "Classifier-Only (No LLM Fusion)",
            "Full Pipeline (Multi-Stage LLM)",
        ]

        results = []
        for approach in approaches:
            result = self.benchmark_approach(approach, test_reviews, test_labels)
            results.append(result)

        winner = max(results, key=lambda x: x["f1"])["approach"] if results else None

        return {
            "test_size": len(test_reviews),
            "results": results,
            "winner": winner,
        }

    @staticmethod
    def pretty_print_benchmark(benchmark_results: Dict[str, Any]) -> None:
        """
        Pretty-print benchmark results.
        """
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS")
        print("=" * 100)

        results = benchmark_results["results"]

        print(
            f"\n{'Approach':<40} "
            f"{'Accuracy':<12} "
            f"{'Precision':<12} "
            f"{'Recall':<12} "
            f"{'F1':<12} "
            f"{'Time (s)':<12} "
            f"{'Uncertain':<12}"
        )
        print("-" * 112)

        for r in results:
            print(
                f"{r['approach']:<40} "
                f"{r['accuracy']:<12.4f} "
                f"{r['precision']:<12.4f} "
                f"{r['recall']:<12.4f} "
                f"{r['f1']:<12.4f} "
                f"{r['avg_time_per_review']:<12.4f} "
                f"{r['uncertain_count']:<12}"
            )

        print("\n" + "=" * 100)
        print(f"BEST APPROACH ON THIS TEST SET: {benchmark_results['winner']}")
        print("=" * 100)

        baseline = next(r for r in results if "Baseline" in r["approach"])
        full = next(r for r in results if "Full Pipeline" in r["approach"])

        print("\nKEY INSIGHTS:")
        if baseline["f1"] > 0:
            f1_improvement = ((full["f1"] - baseline["f1"]) / baseline["f1"]) * 100
            print(f"F1 difference vs baseline: {f1_improvement:+.1f}%")
        else:
            print("Baseline F1 was 0.0, so percentage difference is undefined.")

        print(f"Full pipeline avg time: {full['avg_time_per_review']:.2f}s per review")
        print(f"Full pipeline uncertain outputs: {full['uncertain_count']}")

        if benchmark_results["test_size"] < 20:
            print(
                "\nCAUTION: This benchmark uses a very small test set. "
                "Treat these results as a smoke test, not a reliable performance comparison."
            )


if __name__ == "__main__":
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY")
        raise SystemExit(1)

    # Small smoke-test benchmark only
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
    print("- whether the full pipeline produces uncertain cases")
    print("\nFor a meaningful benchmark, run this on a larger labeled review set.\n")
