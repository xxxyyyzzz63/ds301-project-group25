from __future__ import annotations

import json
import os
import sys

from src.component_validator import LLMComponentValidator


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)

    validator = LLMComponentValidator()

    review = "stayed here last week. wifi terrible. breakfast meh but location good"
    validation = validator.run_full_validation(review, n_consistency_runs=3)
    validator.pretty_print_validation(validation)

    pipeline_summary = validator.validate_saved_pipeline_outputs(
        "outputs/pipeline_test_results.csv"
    )
    validator.pretty_print_pipeline_validation(pipeline_summary)

    out = {
        "single_review_validation": validation,
        "saved_pipeline_validation": pipeline_summary,
    }

    save_path = "outputs/component_validation_summary.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved component validation summary to: {save_path}")


if __name__ == "__main__":
    main()