from __future__ import annotations

import json
import os
import sys

from src.disagreement_analyzer import DisagreementAnalyzer


def main() -> None:
    csv_path = "outputs/pipeline_test_results.csv"
    save_path = "outputs/disagreement_summary.json"

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run test_ai_and_human.py first.")
        sys.exit(1)

    analyzer = DisagreementAnalyzer()
    summary = analyzer.analyze_results_csv(
        csv_path=csv_path,
        only_flagged_cases=True,
    )

    analyzer.pretty_print_summary(summary, n_examples=5)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved disagreement summary to: {save_path}")


if __name__ == "__main__":
    main()