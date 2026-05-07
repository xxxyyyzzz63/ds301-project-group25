from __future__ import annotations

import json
import os
from datetime import datetime

from src.pipeline import ReviewDetectionPipeline


OUTPUT_PATH = "outputs/demo_runs.json"


def save_result(result: dict, output_path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "result": result,
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    pipeline = ReviewDetectionPipeline()

    print("=" * 100)
    print("AI-Generated Hotel Review Detector Demo")
    print("Paste a hotel review below. Press Enter twice when finished.")
    print("=" * 100)

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "":
            break
        lines.append(line)

    review_text = "\n".join(lines).strip()

    if not review_text:
        print("No review text was provided.")
        return

    result = pipeline.run(review_text)

    print("\n" + "=" * 100)
    print("FULL PIPELINE OUTPUT")
    print("=" * 100)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print("Review:")
    print(review_text)
    print()

    classifier_output = result["classifier_output"]
    linguistic_analysis = result["linguistic_analysis"]
    fusion_output = result["fusion_output"]

    print("Classifier label:", classifier_output["predicted_label"])
    print("Classifier AI probability:", classifier_output["ai_probability"])
    print("Classifier AI-likeness score:", classifier_output["ai_likeness_score"])
    print("Classifier uncertainty band:", classifier_output["uncertainty_band"])
    print("Top stylometric features:", classifier_output["top_features"])
    print()

    print("LLM linguistic analysis:")
    print(json.dumps(linguistic_analysis, indent=2, ensure_ascii=False))
    print()

    print("Final fused decision:", fusion_output["final_predicted_label"])
    print("Final uncertainty band:", fusion_output["final_uncertainty_band"])
    print("Agreement status:", fusion_output["agreement_status"])
    print("Final explanation:", fusion_output["final_explanation"])
    print("=" * 100)

    save_result(result, OUTPUT_PATH)
    print(f"\nSaved demo run to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
