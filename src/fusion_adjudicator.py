from __future__ import annotations

import json
import os
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class FusionDecisionOutput(BaseModel):
    """
    Final fused decision combining classifier evidence and linguistic analysis.
    """

    classifier_label: Literal["Human", "AI"] = Field(
        description="Classifier-only prediction"
    )
    classifier_probability: float = Field(
        ge=0.0, le=1.0, description="Classifier AI probability"
    )
    agreement_status: Literal["agree", "mixed", "disagree"] = Field(
        description="Relationship between classifier and linguistic evidence"
    )
    final_predicted_label: Literal["Human", "AI", "Uncertain"] = Field(
        description="Final fused label"
    )
    final_uncertainty_band: Literal[
        "likely human-written",
        "uncertain",
        "likely AI-generated",
    ] = Field(description="Final uncertainty band")
    final_explanation: str = Field(
        description="Explanation referencing both classifier and linguistic evidence"
    )


FUSION_PROMPT = PromptTemplate(
    input_variables=[
        "review_text",
        "classifier_output",
        "linguistic_analysis",
        "forced_agreement_status",
        "forced_final_label",
        "forced_uncertainty_band",
        "rule_notes",
    ],
    template="""
You are the final adjudicator in a multi-stage AI-review detection pipeline.

You are given:
1. The original hotel review
2. The classifier output based on stylometric features
3. A structured linguistic analysis of the review
4. Some rule-based guidance that must be respected

Your job is to combine the classifier evidence and the linguistic-analysis evidence
into a final structured decision.

Rules:
- Do not ignore the classifier output
- Do not ignore the linguistic analysis
- Respect the forced guidance fields below
- If the evidence strongly conflicts, the final label should be Uncertain
- The explanation must reference both classifier evidence and linguistic evidence
- Return valid JSON only
- Do not include markdown fences

Required JSON keys:
classifier_label
classifier_probability
agreement_status
final_predicted_label
final_uncertainty_band
final_explanation

Forced agreement status:
{forced_agreement_status}

Forced final label:
{forced_final_label}

Forced uncertainty band:
{forced_uncertainty_band}

Rule notes:
{rule_notes}

Review:
{review_text}

Classifier output:
{classifier_output}

Linguistic analysis:
{linguistic_analysis}
""".strip(),
)


class FusionAdjudicator:
    """
    Fuses stylometric/classifier evidence with LLM linguistic analysis.
    Includes rule-based constraints so both evidence sources are considered.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before using FusionAdjudicator."
            )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
        )

    @staticmethod
    def _clean_response_text(text: str) -> str:
        """Remove markdown fences if present."""
        text = text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "", 1).strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        elif text.startswith("```"):
            text = text.replace("```", "", 1).strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        return text

    @staticmethod
    def _derive_rule_guidance(
        classifier_output: Dict[str, Any],
        linguistic_analysis: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Derive rule-based guidance to constrain LLM fusion.
        """
        clf_label = classifier_output["predicted_label"]
        clf_prob = float(classifier_output["ai_probability"])

        specificity = linguistic_analysis.get("specificity", "mixed")
        personal = linguistic_analysis.get("personal_experience_markers", "moderate")
        templated = linguistic_analysis.get("templated_language", "moderate")
        messiness = linguistic_analysis.get("human_messiness", "moderate")
        tone = linguistic_analysis.get("tone", "mixed")
        flow = linguistic_analysis.get("narrative_flow", "mixed")

        strongly_human_linguistics = (
            specificity == "concrete"
            and personal == "strong"
            and messiness in {"moderate", "high"}
            and templated == "low"
        )

        strongly_ai_linguistics = (
            specificity in {"generic", "mixed"}
            and personal in {"weak", "moderate"}
            and templated in {"moderate", "high"}
            and messiness == "low"
            and tone in {"polished", "mixed"}
            and flow in {"formulaic", "mixed"}
        )

        forced_agreement_status = "mixed"
        forced_final_label = "Uncertain"
        forced_uncertainty_band = "uncertain"
        rule_notes = []

        if clf_label == "AI" and clf_prob >= 0.90 and strongly_human_linguistics:
            forced_agreement_status = "disagree"
            forced_final_label = "Uncertain"
            forced_uncertainty_band = "uncertain"
            rule_notes.append(
                "Classifier is highly confident AI, but linguistic analysis shows strong human signals."
            )
        elif clf_label == "Human" and clf_prob <= 0.10 and strongly_ai_linguistics:
            forced_agreement_status = "disagree"
            forced_final_label = "Uncertain"
            forced_uncertainty_band = "uncertain"
            rule_notes.append(
                "Classifier is highly confident Human, but linguistic analysis shows strong AI signals."
            )
        elif clf_label == "AI" and strongly_ai_linguistics:
            forced_agreement_status = "agree"
            forced_final_label = "AI"
            forced_uncertainty_band = "likely AI-generated"
            rule_notes.append(
                "Classifier and linguistic analysis both support AI authorship."
            )
        elif clf_label == "Human" and strongly_human_linguistics:
            forced_agreement_status = "agree"
            forced_final_label = "Human"
            forced_uncertainty_band = "likely human-written"
            rule_notes.append(
                "Classifier and linguistic analysis both support human authorship."
            )
        else:
            forced_agreement_status = "mixed"
            forced_final_label = "Uncertain"
            forced_uncertainty_band = "uncertain"
            rule_notes.append(
                "Evidence is mixed, so the final decision should remain cautious."
            )

        return {
            "forced_agreement_status": forced_agreement_status,
            "forced_final_label": forced_final_label,
            "forced_uncertainty_band": forced_uncertainty_band,
            "rule_notes": " ".join(rule_notes),
        }

    def adjudicate(
        self,
        review_text: str,
        classifier_output: Dict[str, Any],
        linguistic_analysis: Dict[str, Any],
    ) -> dict:
        """
        Adjudicate a final decision by fusing classifier and linguistic evidence.
        """
        if not review_text or not review_text.strip():
            raise ValueError("Review text cannot be empty")

        guidance = self._derive_rule_guidance(
            classifier_output=classifier_output,
            linguistic_analysis=linguistic_analysis,
        )

        prompt = FUSION_PROMPT.format(
            review_text=review_text,
            classifier_output=json.dumps(classifier_output, ensure_ascii=False),
            linguistic_analysis=json.dumps(linguistic_analysis, ensure_ascii=False),
            forced_agreement_status=guidance["forced_agreement_status"],
            forced_final_label=guidance["forced_final_label"],
            forced_uncertainty_band=guidance["forced_uncertainty_band"],
            rule_notes=guidance["rule_notes"],
        )

        response = self.llm.invoke(prompt)
        raw_text = self._clean_response_text(response.content)

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse fusion decision JSON.\nRaw response:\n{raw_text}"
            ) from exc

        validated = FusionDecisionOutput(**parsed)
        return validated.model_dump()


if __name__ == "__main__":
    import sys

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with:")
        print("  export OPENAI_API_KEY='your-key-here'   # Mac/Linux")
        print("  set OPENAI_API_KEY=your-key-here        # Windows")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Testing Fusion Adjudicator")
    print("=" * 80)

    review = "The hotel was excellent with impeccable service."

    classifier_output = {
        "predicted_label": "AI",
        "ai_probability": 0.85,
        "ai_likeness_score": 85,
        "uncertainty_band": "likely AI-generated",
        "top_features": {
            "avg_sentence_length": 34.2,
            "templated_language_proxy": 0.0,
        },
    }

    linguistic_analysis = {
        "tone": "polished",
        "specificity": "generic",
        "personal_experience_markers": "weak",
        "templated_language": "high",
        "human_messiness": "low",
        "narrative_flow": "formulaic",
        "evidence_spans": ["excellent with impeccable service"],
        "overall_linguistic_assessment": "Highly polished and generic with limited personal detail.",
    }

    adjudicator = FusionAdjudicator()

    try:
        result = adjudicator.adjudicate(
            review_text=review,
            classifier_output=classifier_output,
            linguistic_analysis=linguistic_analysis,
        )

        print("\nAdjudication returned successfully.")
        print(f"Final label: {result['final_predicted_label']}")
        print(f"Agreement: {result['agreement_status']}")
        print(f"Uncertainty band: {result['final_uncertainty_band']}")
        print(f"Explanation: {result['final_explanation'][:100]}...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Finished test run for FusionAdjudicator.")
    print("=" * 80 + "\n")
