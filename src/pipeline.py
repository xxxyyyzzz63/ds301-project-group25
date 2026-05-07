from __future__ import annotations

from typing import Any, Dict

from src.final_detector import FinalReviewDetector
from src.llm_linguistic_analyzer import LLMLinguisticAnalyzer
from src.fusion_adjudicator import FusionAdjudicator


class ReviewDetectionPipeline:
    """
    Multi-stage LLM-guided review-analysis pipeline.

    Stage 1: LLM linguistic analysis
        - Extracts structured semantic signals from the review

    Stage 2: Stylometric detector
        - Provides calibrated quantitative evidence from stylometric features

    Stage 3: LLM fusion adjudication
        - Combines linguistic signals and classifier evidence into a final decision
    """

    def __init__(self) -> None:
        """
        Initialize the pipeline components.
        """
        self.linguistic_analyzer = LLMLinguisticAnalyzer()
        self.detector = FinalReviewDetector()
        self.adjudicator = FusionAdjudicator()

    def run(self, review_text: str) -> Dict[str, Any]:
        """
        Execute the full pipeline.

        Returns a dictionary containing:
        - review_text
        - linguistic_analysis
        - classifier_output
        - fusion_output
        """
        if not review_text or not review_text.strip():
            raise ValueError("Review text cannot be empty")

        # Stage 1: LLM linguistic analysis
        linguistic_analysis = self.linguistic_analyzer.analyze(review_text)

        # Stage 2: Stylometric classifier evidence
        classifier_output = self.detector.detect(review_text)

        # Stage 3: LLM fusion adjudication
        fusion_output = self.adjudicator.adjudicate(
            review_text=review_text,
            classifier_output=classifier_output,
            linguistic_analysis=linguistic_analysis,
        )

        return {
            "review_text": review_text,
            "linguistic_analysis": linguistic_analysis,
            "classifier_output": classifier_output,
            "fusion_output": fusion_output,
        }

    @staticmethod
    def pretty_print(result: Dict[str, Any]) -> None:
        """
        Pretty-print the full pipeline output.
        """
        print("\n" + "=" * 100)
        print("INPUT REVIEW")
        print("=" * 100)
        print(result["review_text"])
        print()

        print("=" * 100)
        print("STAGE 1: LLM LINGUISTIC ANALYSIS")
        print("=" * 100)
        ling = result["linguistic_analysis"]
        print(f"Tone:                      {ling['tone']}")
        print(f"Specificity:               {ling['specificity']}")
        print(f"Personal Experience:       {ling['personal_experience_markers']}")
        print(f"Templated Language:        {ling['templated_language']}")
        print(f"Human Messiness:           {ling['human_messiness']}")
        print(f"Narrative Flow:            {ling['narrative_flow']}")
        print(f"\nEvidence Spans:            {ling['evidence_spans']}")
        print("\nLinguistic Assessment:")
        print(f"  {ling['overall_linguistic_assessment']}")
        print()

        print("=" * 100)
        print("STAGE 2: STYLOMETRIC DETECTOR")
        print("=" * 100)
        clf = result["classifier_output"]
        print(f"Model:                     {clf['model_used']}")
        print(f"Calibrated:                {clf['calibrated']}")
        print(f"Predicted Label:           {clf['predicted_label']}")
        print(f"AI Probability:            {clf['ai_probability']:.4f}")
        print(f"AI-Likeness Score:         {clf['ai_likeness_score']}/100")
        print(f"Uncertainty Band:          {clf['uncertainty_band']}")
        print("\nTop Discriminative Features:")
        for feature, value in list(clf["top_features"].items())[:5]:
            print(f"  {feature:<30} {value:.4f}")
        print()

        print("=" * 100)
        print("STAGE 3: LLM FUSION ADJUDICATION")
        print("=" * 100)
        fusion = result["fusion_output"]
        print(f"Classifier Label:          {fusion['classifier_label']}")
        print(f"Classifier Probability:    {fusion['classifier_probability']:.4f}")
        print(f"Agreement Status:          {fusion['agreement_status']}")
        print(f"\nFinal Predicted Label:     {fusion['final_predicted_label']}")
        print(f"Final Uncertainty Band:    {fusion['final_uncertainty_band']}")
        print("\nGrounded Explanation:")
        print(f"  {fusion['final_explanation']}")
        print()

        print("=" * 100)
        print("PIPELINE SUMMARY")
        print("=" * 100)
        print(
            f"Stage 1: {ling['tone']} tone, {ling['specificity']} specificity"
        )
        print(
            f"Stage 2: {clf['predicted_label']} ({clf['ai_probability']:.2%} AI probability)"
        )
        print(
            f"Stage 3: {fusion['final_predicted_label']} ({fusion['agreement_status']} between stages)"
        )
        print("=" * 100 + "\n")


if __name__ == "__main__":
    import os
    import sys
    import traceback

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with:")
        print("  export OPENAI_API_KEY='your-key-here'   # Mac/Linux")
        print("  set OPENAI_API_KEY=your-key-here        # Windows")
        sys.exit(1)

    print("\n" + "=" * 100)
    print("MULTI-STAGE REVIEW DETECTION PIPELINE DEMO")
    print("=" * 100)
    print("This pipeline uses:")
    print("  1. LLM linguistic analysis")
    print("  2. Stylometric classifier evidence")
    print("  3. LLM fusion adjudication")
    print()

    test_reviews = [
        "The room was clean and the location was convenient, but the breakfast was disappointing and the walls were thin.",
        "This establishment exceeded all expectations with impeccable service and world-class amenities.",
    ]

    pipeline = ReviewDetectionPipeline()

    for i, review in enumerate(test_reviews, 1):
        print("\n" + "=" * 100)
        print(f"TEST CASE {i}")
        print("=" * 100)

        try:
            result = pipeline.run(review)
            ReviewDetectionPipeline.pretty_print(result)
            print("Finished pipeline run for this test case.")
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "=" * 100)
    print("Finished test runs for ReviewDetectionPipeline.")
    print("=" * 100 + "\n")
