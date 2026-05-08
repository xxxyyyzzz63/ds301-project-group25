from __future__ import annotations

import json
import os
import sys

from src.attribution_analyzer import AttributionAnalyzer


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    csv_path = "outputs/pipeline_test_results.csv"
    save_path = "outputs/attribution_analysis_results.json"

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run test_ai_and_human.py first.")
        sys.exit(1)

    analyzer = AttributionAnalyzer()
    analyses = analyzer.analyze_cases_from_csv(
        csv_path=csv_path,
        max_cases=5,
        filter_mode="flagged",
    )

    print("\n" + "=" * 80)
    print("ATTRIBUTION ANALYSIS RESULTS")
    print("=" * 80)

    for i, analysis in enumerate(analyses, start=1):
        print(f"\nCASE {i}")
        analyzer.pretty_print_attribution(analysis)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(analyses, f, indent=2, ensure_ascii=False)

    print(f"\nSaved attribution analyses to: {save_path}")


if __name__ == "__main__":
    main()