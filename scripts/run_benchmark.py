from __future__ import annotations

import json
import os
import sys
from datetime import datetime, UTC
from typing import Tuple

import pandas as pd

from src.benchmark import LLMBenchmark


AI_PATH = "data/ai_generated_tripadvisor_reviews_openai_diverse.csv"
HUMAN_PATH = "data/tripadvisor_hotel_reviews.csv"
OUTPUT_DIR = "outputs"


def load_balanced_sample(
    ai_path: str,
    human_path: str,
    n_ai: int,
    n_human: int,
    random_state: int = 42,
) -> Tuple[list[str], list[str]]:
    """
    Load a balanced labeled sample from the current project datasets.
    """
    ai_df = pd.read_csv(ai_path)
    human_df = pd.read_csv(human_path)

    if "Review" not in ai_df.columns:
        raise ValueError(f"AI dataset at {ai_path} must contain a 'Review' column.")
    if "Review" not in human_df.columns:
        raise ValueError(f"Human dataset at {human_path} must contain a 'Review' column.")

    n_ai = min(n_ai, len(ai_df))
    n_human = min(n_human, len(human_df))

    ai_sample = ai_df.sample(n=n_ai, random_state=random_state).copy()
    human_sample = human_df.sample(n=n_human, random_state=random_state).copy()

    ai_sample["label"] = "AI"
    human_sample["label"] = "Human"

    combined = pd.concat([ai_sample, human_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

    reviews = combined["Review"].astype(str).tolist()
    labels = combined["label"].astype(str).tolist()

    return reviews, labels


def save_benchmark_results(
    benchmark_results: dict,
    n_ai: int,
    n_human: int,
    random_state: int,
) -> tuple[str, str]:
    """
    Save benchmark results to JSON and CSV for later reporting.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(
        OUTPUT_DIR,
        f"benchmark_results_{n_ai}ai_{n_human}human_seed{random_state}_{timestamp}.json",
    )
    csv_path = os.path.join(
        OUTPUT_DIR,
        f"benchmark_results_{n_ai}ai_{n_human}human_seed{random_state}_{timestamp}.csv",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

    rows = []
    for r in benchmark_results["results"]:
        rows.append({
            "approach": r["approach"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "avg_time_per_review": r["avg_time_per_review"],
            "total_time": r["total_time"],
            "uncertain_count": r["uncertain_count"],
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return json_path, csv_path


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY")
        sys.exit(1)

    n_ai = 10
    n_human = 10
    random_state = 42

    if len(sys.argv) >= 2:
        n_ai = int(sys.argv[1])
    if len(sys.argv) >= 3:
        n_human = int(sys.argv[2])
    if len(sys.argv) >= 4:
        random_state = int(sys.argv[3])

    print("\n" + "=" * 100)
    print("RUNNING BENCHMARK ON CURRENT PROJECT DATASETS")
    print("=" * 100)
    print(f"AI dataset: {AI_PATH}")
    print(f"Human dataset: {HUMAN_PATH}")
    print(f"Requested sample: {n_ai} AI + {n_human} Human")
    print(f"Random state: {random_state}")

    reviews, labels = load_balanced_sample(
        ai_path=AI_PATH,
        human_path=HUMAN_PATH,
        n_ai=n_ai,
        n_human=n_human,
        random_state=random_state,
    )

    print(f"\nLoaded {len(reviews)} total reviews.")
    print(f"AI labels: {sum(1 for x in labels if x == 'AI')}")
    print(f"Human labels: {sum(1 for x in labels if x == 'Human')}")

    benchmark = LLMBenchmark()
    results = benchmark.run_full_benchmark(reviews, labels)
    benchmark.pretty_print_benchmark(results)

    json_path, csv_path = save_benchmark_results(
        benchmark_results=results,
        n_ai=n_ai,
        n_human=n_human,
        random_state=random_state,
    )

    print(f"\nSaved benchmark JSON to: {json_path}")
    print(f"Saved benchmark CSV to:  {csv_path}")
    print("\nFinished benchmark run.\n")


if __name__ == "__main__":
    main()
