from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class AttributionAnalysis(BaseModel):
    """
    Explains which linguistic dimensions contributed most to the final decision.
    """

    primary_signal: str = Field(description="Most important linguistic dimension")
    primary_explanation: str = Field(description="Why this dimension was decisive")
    supporting_signals: List[str] = Field(default_factory=list)
    conflicting_signals: List[str] = Field(default_factory=list)
    linguistic_contribution_score: int = Field(
        ge=0,
        le=100,
        description="How much linguistic evidence contributed relative to stylometric evidence"
    )
    key_evidence_span: str = Field(description="Most telling quoted phrase")
    counterfactual_analysis: str = Field(
        description="What would need to change linguistically to flip the decision"
    )


ATTRIBUTION_PROMPT = PromptTemplate(
    input_variables=[
        "review_text",
        "true_label",
        "linguistic_analysis",
        "classifier_output",
        "fusion_output",
    ],
    template="""
You are analyzing WHICH linguistic dimensions contributed most to a final AI-vs-human review decision.

You are given:
- the original review
- the true label, if available
- structured linguistic analysis
- classifier output
- final fusion output

Your task is attribution analysis.

Use only these six linguistic dimensions as signal names:
- tone
- specificity
- personal_experience_markers
- templated_language
- human_messiness
- narrative_flow

Rules:
- primary_signal must be exactly one of those six names
- supporting_signals must only contain names from those six dimensions
- conflicting_signals must only contain names from those six dimensions, or classifier-based conflicts like "classifier_ai_probability" or "classifier_ai_likeness_score"
- linguistic_contribution_score must be an INTEGER from 0 to 100
- return valid JSON only
- do not include markdown fences
- use concise but specific explanations
- if true_label is provided, you may mention whether the attribution seems consistent with the ground truth

Return JSON with exactly these keys:
primary_signal
primary_explanation
supporting_signals
conflicting_signals
linguistic_contribution_score
key_evidence_span
counterfactual_analysis

Review:
{review_text}

True label:
{true_label}

Linguistic analysis:
{linguistic_analysis}

Classifier output:
{classifier_output}

Fusion output:
{fusion_output}
""".strip(),
)


class AttributionAnalyzer:
    """
    LLM-based attribution analyzer for explaining which linguistic dimensions mattered most.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before using AttributionAnalyzer."
            )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
        )

    @staticmethod
    def _clean_response_text(text: str) -> str:
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
    def _normalize_score(value: Any) -> int:
        if value is None:
            return 50

        if isinstance(value, str):
            value = value.strip()
            try:
                value = float(value)
            except ValueError:
                return 50

        if isinstance(value, (int, float)):
            if 0 <= float(value) <= 1:
                value = float(value) * 100
            value = int(round(float(value)))
            return max(0, min(100, value))

        return 50

    @staticmethod
    def _normalize_signal_list(values: Any) -> List[str]:
        allowed = {
            "tone",
            "specificity",
            "personal_experience_markers",
            "templated_language",
            "human_messiness",
            "narrative_flow",
            "classifier_ai_probability",
            "classifier_ai_likeness_score",
        }

        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        if not isinstance(values, list):
            return []

        cleaned = []
        for v in values:
            if not isinstance(v, str):
                continue
            v = v.strip()
            if v in allowed:
                cleaned.append(v)
            else:
                # Simple normalization of common variants
                mapping = {
                    "personal markers": "personal_experience_markers",
                    "personal_experience": "personal_experience_markers",
                    "templated language": "templated_language",
                    "human messiness": "human_messiness",
                    "narrative flow": "narrative_flow",
                    "ai probability": "classifier_ai_probability",
                    "ai-likeness score": "classifier_ai_likeness_score",
                }
                normalized = mapping.get(v.lower())
                if normalized:
                    cleaned.append(normalized)

        # preserve order while deduplicating
        seen = set()
        deduped = []
        for item in cleaned:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    @staticmethod
    def _normalize_primary_signal(value: Any) -> str:
        allowed = {
            "tone",
            "specificity",
            "personal_experience_markers",
            "templated_language",
            "human_messiness",
            "narrative_flow",
        }

        if not isinstance(value, str):
            return "tone"

        value = value.strip()
        if value in allowed:
            return value

        mapping = {
            "personal markers": "personal_experience_markers",
            "personal_experience": "personal_experience_markers",
            "templated language": "templated_language",
            "human messiness": "human_messiness",
            "narrative flow": "narrative_flow",
        }
        return mapping.get(value.lower(), "tone")

    @staticmethod
    def _normalize_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(parsed)

        normalized["linguistic_contribution_score"] = AttributionAnalyzer._normalize_score(
            normalized.get("linguistic_contribution_score")
        )

        normalized["primary_signal"] = AttributionAnalyzer._normalize_primary_signal(
            normalized.get("primary_signal")
        )

        normalized["supporting_signals"] = AttributionAnalyzer._normalize_signal_list(
            normalized.get("supporting_signals")
        )

        normalized["conflicting_signals"] = AttributionAnalyzer._normalize_signal_list(
            normalized.get("conflicting_signals")
        )

        if "primary_explanation" not in normalized or normalized["primary_explanation"] is None:
            normalized["primary_explanation"] = ""
        if "key_evidence_span" not in normalized or normalized["key_evidence_span"] is None:
            normalized["key_evidence_span"] = ""
        if "counterfactual_analysis" not in normalized or normalized["counterfactual_analysis"] is None:
            normalized["counterfactual_analysis"] = ""

        return normalized

    @staticmethod
    def _row_to_pipeline_like_dict(row: pd.Series) -> Dict[str, Any]:
        evidence_spans = []
        top_features = {}

        if pd.notna(row.get("ling_evidence_spans")):
            try:
                evidence_spans = json.loads(row["ling_evidence_spans"])
            except Exception:
                evidence_spans = []

        if pd.notna(row.get("classifier_top_features")):
            try:
                top_features = json.loads(row["classifier_top_features"])
            except Exception:
                top_features = {}

        return {
            "review_text": row.get("review_text"),
            "true_label": row.get("true_label"),
            "linguistic_analysis": {
                "tone": row.get("ling_tone"),
                "specificity": row.get("ling_specificity"),
                "personal_experience_markers": row.get("ling_personal_experience_markers"),
                "templated_language": row.get("ling_templated_language"),
                "human_messiness": row.get("ling_human_messiness"),
                "narrative_flow": row.get("ling_narrative_flow"),
                "evidence_spans": evidence_spans,
                "overall_linguistic_assessment": row.get("ling_overall_assessment"),
            },
            "classifier_output": {
                "predicted_label": row.get("classifier_predicted_label"),
                "ai_probability": row.get("classifier_ai_probability"),
                "ai_likeness_score": row.get("classifier_ai_likeness_score"),
                "uncertainty_band": row.get("classifier_uncertainty_band"),
                "top_features": top_features,
                "explanation": row.get("classifier_explanation"),
            },
            "fusion_output": {
                "classifier_label": row.get("fusion_classifier_label"),
                "classifier_probability": row.get("fusion_classifier_probability"),
                "agreement_status": row.get("fusion_agreement_status"),
                "final_predicted_label": row.get("fusion_final_predicted_label"),
                "final_uncertainty_band": row.get("fusion_final_uncertainty_band"),
                "final_explanation": row.get("fusion_final_explanation"),
            },
        }

    def analyze_single_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ATTRIBUTION_PROMPT.format(
            review_text=case["review_text"],
            true_label=case.get("true_label"),
            linguistic_analysis=json.dumps(case["linguistic_analysis"], ensure_ascii=False),
            classifier_output=json.dumps(case["classifier_output"], ensure_ascii=False),
            fusion_output=json.dumps(case["fusion_output"], ensure_ascii=False),
        )

        response = self.llm.invoke(prompt)
        raw_text = self._clean_response_text(response.content)

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse attribution JSON.\nRaw response:\n{raw_text}"
            ) from exc

        normalized = self._normalize_output(parsed)
        validated = AttributionAnalysis(**normalized)

        result = validated.model_dump()
        result["review_text"] = case["review_text"]
        result["true_label"] = case.get("true_label")
        result["final_predicted_label"] = case["fusion_output"]["final_predicted_label"]
        result["agreement_status"] = case["fusion_output"]["agreement_status"]
        return result

    def analyze_cases_from_csv(
        self,
        csv_path: str,
        max_cases: int = 5,
        filter_mode: str = "flagged"
    ) -> List[Dict[str, Any]]:
        df = pd.read_csv(csv_path)

        if filter_mode == "flagged":
            df = df[
                (df["fusion_agreement_status"] == "disagree")
                | (df["fusion_final_predicted_label"] == "Uncertain")
            ]
        elif filter_mode == "ai_only":
            df = df[df["true_label"] == "AI"]
        elif filter_mode == "human_only":
            df = df[df["true_label"] == "Human"]
        elif filter_mode == "all":
            pass
        else:
            raise ValueError(f"Unknown filter_mode: {filter_mode}")

        df = df.head(max_cases).copy()

        analyses = []
        for _, row in df.iterrows():
            case = self._row_to_pipeline_like_dict(row)
            analyses.append(self.analyze_single_case(case))

        return analyses

    @staticmethod
    def pretty_print_attribution(attribution: Dict[str, Any]) -> None:
        print("\n" + "=" * 100)
        print("LINGUISTIC ATTRIBUTION ANALYSIS")
        print("=" * 100)

        print(f"\nTrue label: {attribution.get('true_label')}")
        print(f"Final predicted label: {attribution.get('final_predicted_label')}")
        print(f"Agreement status: {attribution.get('agreement_status')}")
        print(f"\nReview: {str(attribution.get('review_text'))[:220]}...")

        print(f"\nPrimary signal: {attribution['primary_signal']}")
        print(f"Primary explanation: {attribution['primary_explanation']}")

        if attribution["supporting_signals"]:
            print(f"\nSupporting signals: {', '.join(attribution['supporting_signals'])}")

        if attribution["conflicting_signals"]:
            print(f"Conflicting signals: {', '.join(attribution['conflicting_signals'])}")

        print(f"\nLinguistic contribution score: {attribution['linguistic_contribution_score']}/100")
        print(f"Key evidence span: \"{attribution['key_evidence_span']}\"")
        print(f"Counterfactual analysis: {attribution['counterfactual_analysis']}")
        print("=" * 100)


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 80)
    print("TESTING: Attribution Analyzer")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    csv_path = "outputs/pipeline_test_results.csv"
    if not os.path.exists(csv_path):
        print(f"\nERROR: {csv_path} not found.")
        print("Run test_ai_and_human.py first.")
        sys.exit(1)

    analyzer = AttributionAnalyzer()
    analyses = analyzer.analyze_cases_from_csv(
        csv_path=csv_path,
        max_cases=2,
        filter_mode="flagged",
    )

    for analysis in analyses:
        analyzer.pretty_print_attribution(analysis)

    print("\nFinished test run for AttributionAnalyzer.\n")