"""
Baseline AI-Generated Text Detector
Single-pass LLM approach with structured outputs
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from typing import Literal
import os


# Pydantic schema for structured output
class AIDetectionResult(BaseModel):
    """Structured output for AI detection"""
    prediction: Literal["human", "ai"] = Field(
        description="Whether the review is human-written or AI-generated"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Brief explanation for the prediction"
    )


# Few-shot examples (following Alshammari & Rao 2023 methodology)
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


# Prompt template
DETECTION_PROMPT = PromptTemplate(
    input_variables=["review"],
    template="""You are an expert at detecting AI-generated hotel reviews.

{few_shot_examples}

Now analyze this review and determine if it's human-written or AI-generated.

Review: {review}

Provide your analysis in the following format:
Prediction: [human or ai]
Confidence: [0.0 to 1.0]
Reasoning: [brief explanation]
"""
)


class BaselineDetector:
    """Single-pass LLM detector for AI-generated reviews"""
    
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.0):
        """Initialize the detector with OpenAI model"""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=DETECTION_PROMPT
        )
    
    def detect(self, review: str) -> AIDetectionResult:
        """
        Detect if a review is AI-generated
        
        Args:
            review: Hotel review text
            
        Returns:
            AIDetectionResult with prediction, confidence, and reasoning
        """
        # Run the chain
        result = self.chain.run(
            review=review,
            few_shot_examples=FEW_SHOT_EXAMPLES
        )
        
        # Parse the result (simple parsing for baseline)
        lines = result.strip().split('\n')
        prediction = "ai"
        confidence = 0.5
        reasoning = ""
        
        for line in lines:
            if line.startswith("Prediction:"):
                pred_text = line.split(":", 1)[1].strip().lower()
                prediction = "ai" if "ai" in pred_text else "human"
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    confidence = 0.5
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return AIDetectionResult(
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def predict_batch(self, reviews: list) -> list:
        """Predict for multiple reviews"""
        return [self.detect(review) for review in reviews]


if __name__ == "__main__":
    # Test the baseline detector
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
