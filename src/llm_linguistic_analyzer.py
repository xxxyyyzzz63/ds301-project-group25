from __future__ import annotations

import json
import os
from typing import List, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class LinguisticAnalysisOutput(BaseModel):
    """
    Structured linguistic analysis output for a hotel review.
    """

    tone: Literal["casual", "polished", "mixed"] = Field(
        description="Overall tone/register of the review"
    )
    specificity: Literal["concrete", "generic", "mixed"] = Field(
        description="Whether the review gives specific details or generic statements"
    )
    personal_experience_markers: Literal["strong", "moderate", "weak"] = Field(
        description="Strength of first-hand or subjective voice"
    )
    templated_language: Literal["high", "moderate", "low"] = Field(
        description="Whether phrasing feels formulaic or templated"
    )
    human_messiness: Literal["high", "moderate", "low"] = Field(
        description="Presence of natural disfluencies, shorthand, or rough edges"
    )
    narrative_flow: Literal["natural", "formulaic", "mixed"] = Field(
        description="Whether the review flows naturally or reads like a template"
    )
    evidence_spans: List[str] = Field(
        default_factory=list,
        description="Short quoted spans from the review supporting the analysis"
    )
    overall_linguistic_assessment: str = Field(
        description="Brief summary of the review's linguistic character"
    )


LINGUISTIC_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["review_text"],
    template="""
You are analyzing a hotel review for linguistic signals relevant to AI-vs-human authorship.

IMPORTANT: Do NOT make the final classification yet.
Your task is to analyze HOW the review is written, not WHO wrote it.

Focus on:
- whether the tone is overly polished or more natural/casual
- whether details are concrete and lived-in versus generic
- whether the language feels templated or repetitive
- whether the review shows human messiness, irregularity, shorthand, or uneven pacing
- whether the review has natural narrative flow or formulaic structure

Rules:
- Base your analysis only on the review text
- Return valid JSON only
- Include 1 to 3 short evidence spans copied exactly from the review when possible
- Do not include markdown fences

Return JSON with exactly these keys:
tone
specificity
personal_experience_markers
templated_language
human_messiness
narrative_flow
evidence_spans
overall_linguistic_assessment

Review:
{review_text}
""".strip(),
)


class LLMLinguisticAnalyzer:
    """
    LLM-based linguistic analyzer for Stage 1 of the pipeline.
    This component produces structured evidence about how the review sounds.
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
                "Set it before using LLMLinguisticAnalyzer."
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

    def analyze(self, review_text: str) -> dict:
        """
        Analyze a review and return structured linguistic dimensions.
        """
        if not review_text or not review_text.strip():
            raise ValueError("Review text cannot be empty")

        prompt = LINGUISTIC_ANALYSIS_PROMPT.format(review_text=review_text)
        response = self.llm.invoke(prompt)
        raw_text = self._clean_response_text(response.content)

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse linguistic analysis JSON.\nRaw response:\n{raw_text}"
            ) from exc

        validated = LinguisticAnalysisOutput(**parsed)
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
    print("Testing LLM Linguistic Analyzer")
    print("=" * 80)

    test_reviews = [
        {
            "text": "stayed here last week. wifi terrible. breakfast meh but location good",
            "expected": "casual tone, concrete details, human messiness",
        },
        {
            "text": "This establishment exceeded all expectations with impeccable service and world-class amenities.",
            "expected": "polished tone, generic language, templated",
        },
    ]

    analyzer = LLMLinguisticAnalyzer()

    for i, review in enumerate(test_reviews, 1):
        print("\n" + "-" * 80)
        print(f"Test Case {i}")
        print("-" * 80)
        print(f"Review: {review['text']}")
        print(f"Expected: {review['expected']}")

        try:
            result = analyzer.analyze(review["text"])
            print("\nAnalysis returned successfully.")
            print(f"Tone: {result['tone']}")
            print(f"Specificity: {result['specificity']}")
            print(f"Templated language: {result['templated_language']}")
            print(f"Evidence spans: {result['evidence_spans']}")
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("Finished test runs for LLMLinguisticAnalyzer.")
    print("=" * 80 + "\n")
