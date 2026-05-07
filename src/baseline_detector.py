"""
Baseline AI-Generated Text Detector
Single-pass LLM approach with structured outputs
"""

from __future__ import annotations

import os
from typing import Literal, List

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field


class AIDetectionResult(BaseModel):
    """Structured output for AI detection."""
    prediction: Literal["human", "ai"] = Field(
        description="Whether the review is human-written or AI-generated"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Brief explanation for the prediction"
    )


FEW_SHOT_EXAMPLES = """
Example 1:
Review: "This hotel exceeded all expectations! The staff went above and beyond to ensure our comfort. Every detail was meticulously attended to, and the amenities were top-notch. I cannot recommend this establishment highly enough for anyone seeking a truly exceptional experience."
Label: AI-generated
Reasoning: Overly formal language, generic superlatives, perfect grammar without natural speech patterns

Example 2:
Review: "stayed here last week. rooms ok but wifi terrible. breakfast was meh. location good tho, walked to downtown easy. would prob stay again if price right"
Label: Human
Reasoning: Casual tone, abbreviations, minor grammar quirks, authentic personal experience

Example 3:
Review: "The hotel provides excellent accommodations with state-of-the-art facilities. The dining experience was remarkable, featuring a diverse array of culinary options. The attentive service staff demonstrated exceptional professionalism throughout our stay."
Label: AI-generated
Reasoning: Corporate tone, lack of specific details, overly polished without personality
"""


DETECTION_PROMPT = PromptTemplate(
    input_variables=["review", "few_shot_examples"],
    template="""You are an expert at detecting AI-generated hotel reviews.

{few_shot_examples}

Now analyze this review and determine if it is human-written or AI-generated.

Review: {review}

Provide your analysis in exactly this format:
Prediction: [human or ai]
Confidence: [0.0 to 1.0]
Reasoning: [brief explanation]
"""
)


class BaselineDetector:
    """Single-pass LLM detector for AI-generated reviews."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """Initialize the detector with an OpenAI chat model."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before using BaselineDetector."
            )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )

        # Modern LangChain pattern: prompt | llm
        self.chain = DETECTION_PROMPT | self.llm

    def _parse_result(self, text: str) -> AIDetectionResult:
        """Parse the raw model text into a structured result."""
        lines = text.strip().split("\n")
        prediction = "ai"
        confidence = 0.5
        reasoning = ""

        for line in lines:
            lower = line.lower().strip()

            if lower.startswith("prediction:"):
                pred_text = line.split(":", 1)[1].strip().lower()
                prediction = "ai" if "ai" in pred_text else "human"

            elif lower.startswith("confidence:"):
                conf_text = line.split(":", 1)[1].strip()
                try:
                    confidence = float(conf_text)
                except ValueError:
                    confidence = 0.5

            elif lower.startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        return AIDetectionResult(
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning
        )

    def detect(self, review: str) -> AIDetectionResult:
        """
        Detect if a review is AI-generated.

        Args:
            review: Hotel review text

        Returns:
            AIDetectionResult with prediction, confidence, and reasoning
        """
        if not review or not review.strip():
            raise ValueError("Review text cannot be empty.")

        result = self.chain.invoke(
            {
                "review": review,
                "few_shot_examples": FEW_SHOT_EXAMPLES,
            }
        )

        raw_text = result.content if hasattr(result, "content") else str(result)
        return self._parse_result(raw_text)

    def predict_batch(self, reviews: List[str]) -> List[AIDetectionResult]:
        """Predict for multiple reviews."""
        return [self.detect(review) for review in reviews]


if __name__ == "__main__":
    detector = BaselineDetector()

    test_review = """
    I recently had the pleasure of staying at this exceptional establishment.
    The accommodations were of the highest quality, and the service staff
    demonstrated remarkable attention to detail. The culinary offerings were
    diverse and expertly prepared. I would highly recommend this venue to
    discerning travelers seeking premium hospitality.
    """

    result = detector.detect(test_review)
    print(f"\nPrediction: {result.prediction}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
