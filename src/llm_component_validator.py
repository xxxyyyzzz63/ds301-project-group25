from __future__ import annotations

from typing import Dict, Any, List
from collections import Counter

from src.llm_linguistic_analyzer import LLMLinguisticAnalyzer


class LLMComponentValidator:
    """
    Validates LLM components through systematic testing.
    
    Tests:
    - Consistency: Same input → similar outputs
    - Grounding: Evidence spans exist in text
    - Schema: Outputs match expected structure
    """
    
    def __init__(self) -> None:
        self.linguistic_analyzer = LLMLinguisticAnalyzer()
    
    def validate_consistency(
        self,
        review_text: str,
        n_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Test consistency by running analysis multiple times.
        At temp=0, outputs should be identical or very similar.
        
        Args:
            review_text: Review to analyze
            n_runs: Number of times to run (default: 3)
            
        Returns:
            Consistency metrics
        """
        print(f"Running consistency test ({n_runs} runs)...")
        
        results = []
        for i in range(n_runs):
            result = self.linguistic_analyzer.analyze(review_text)
            results.append(result)
            print(f"Run {i+1}/{n_runs} complete")
        
        # Check agreement on each dimension
        dimensions = [
            "tone", "specificity", "personal_experience_markers",
            "templated_language", "human_messiness", "narrative_flow"
        ]
        
        agreement_scores = {}
        for dim in dimensions:
            values = [r[dim] for r in results]
            most_common_value, count = Counter(values).most_common(1)[0]
            agreement_pct = (count / n_runs) * 100
            agreement_scores[dim] = {
                "agreement_pct": agreement_pct,
                "most_common": most_common_value,
                "all_values": values
            }
        
        avg_agreement = sum(
            s["agreement_pct"] for s in agreement_scores.values()
        ) / len(dimensions)
        
        return {
            "n_runs": n_runs,
            "overall_consistency": avg_agreement,
            "dimension_agreement": agreement_scores,
            "interpretation": (
                "Excellent (≥90%)" if avg_agreement >= 90
                else "Good (70-90%)" if avg_agreement >= 70
                else "Poor (<70%)"
            )
        }
    
    def validate_evidence_grounding(
        self,
        review_text: str,
        linguistic_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that evidence spans actually exist in review.
        Ensures no hallucination.
        
        Args:
            review_text: Original review
            linguistic_analysis: LLM output
            
        Returns:
            Grounding validation results
        """
        print("Validating evidence grounding...")
        
        evidence_spans = linguistic_analysis.get("evidence_spans", [])
        
        if not evidence_spans:
            return {
                "grounded": True,
                "warning": "No evidence spans provided",
                "missing_spans": []
            }
        
        missing_spans = []
        for span in evidence_spans:
            # Case-insensitive check
            if span.lower() not in review_text.lower():
                missing_spans.append(span)
        
        grounded_pct = (
            ((len(evidence_spans) - len(missing_spans)) / len(evidence_spans)) * 100
        )
        
        return {
            "grounded": len(missing_spans) == 0,
            "grounded_percentage": grounded_pct,
            "total_spans": len(evidence_spans),
            "grounded_spans": len(evidence_spans) - len(missing_spans),
            "missing_spans": missing_spans,
            "interpretation": (
                "All grounded" if len(missing_spans) == 0
                else f"{len(missing_spans)} not found"
            )
        }
    
    def validate_schema_compliance(
        self,
        linguistic_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate output follows expected schema.
        
        Args:
            linguistic_analysis: LLM output
            
        Returns:
            Schema validation results
        """
        print("Validating schema compliance...")
        
        required_fields = [
            "tone", "specificity", "personal_experience_markers",
            "templated_language", "human_messiness", "narrative_flow",
            "evidence_spans", "overall_linguistic_assessment"
        ]
        
        expected_values = {
            "tone": ["casual", "polished", "mixed"],
            "specificity": ["concrete", "generic", "mixed"],
            "personal_experience_markers": ["strong", "moderate", "weak"],
            "templated_language": ["high", "moderate", "low"],
            "human_messiness": ["high", "moderate", "low"],
            "narrative_flow": ["natural", "formulaic", "mixed"]
        }
        
        # Check required fields
        missing_fields = [
            f for f in required_fields if f not in linguistic_analysis
        ]
        
        # Check value validity
        invalid_values = {}
        for field, allowed in expected_values.items():
            if field in linguistic_analysis:
                value = linguistic_analysis[field]
                if value not in allowed:
                    invalid_values[field] = {
                        "actual": value,
                        "expected": allowed
                    }
        
        # Check evidence spans is list
        evidence_valid = isinstance(
            linguistic_analysis.get("evidence_spans", []), list
        )
        
        compliant = (
            len(missing_fields) == 0
            and len(invalid_values) == 0
            and evidence_valid
        )
        
        return {
            "compliant": compliant,
            "missing_fields": missing_fields,
            "invalid_values": invalid_values,
            "evidence_spans_is_list": evidence_valid,
            "interpretation": (
                "Fully compliant" if compliant
                else "Validation failed"
            )
        }
    
    def run_full_validation(
        self,
        review_text: str,
        n_consistency_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Run complete validation suite.
        
        Args:
            review_text: Review to validate
            n_consistency_runs: Runs for consistency test
            
        Returns:
            Complete validation report
        """
        print("\n" + "="*100)
        print("LLM COMPONENT VALIDATION SUITE")
        print("="*100)
        print(f"Review: {review_text[:80]}...\n")
        
        # Get one analysis
        linguistic_analysis = self.linguistic_analyzer.analyze(review_text)
        
        # Run all validations
        consistency_result = self.validate_consistency(
            review_text, n_consistency_runs
        )
        grounding_result = self.validate_evidence_grounding(
            review_text, linguistic_analysis
        )
        schema_result = self.validate_schema_compliance(linguistic_analysis)
        
        # Overall assessment
        all_passed = (
            consistency_result["overall_consistency"] >= 70
            and grounding_result["grounded"]
            and schema_result["compliant"]
        )
        
        return {
            "review_text": review_text,
            "consistency_validation": consistency_result,
            "grounding_validation": grounding_result,
            "schema_validation": schema_result,
            "overall_passed": all_passed,
            "summary": (
                "ALL PASSED - LLM component is systematic"
                if all_passed
                else "SOME FAILED - review component"
            )
        }
    
    @staticmethod
    def pretty_print_validation(validation_result: Dict[str, Any]) -> None:
        """Pretty-print validation results."""
        print("\n" + "="*100)
        print("VALIDATION RESULTS")
        print("="*100)
        
        cons = validation_result["consistency_validation"]
        print(f"CONSISTENCY ({cons['n_runs']} runs)")
        print(f"Overall: {cons['overall_consistency']:.1f}% agreement")
        print(f"{cons['interpretation']}")
        
        ground = validation_result["grounding_validation"]
        print(f"EVIDENCE GROUNDING")
        print(f"{ground['interpretation']}")
        
        schema = validation_result["schema_validation"]
        print(f"SCHEMA COMPLIANCE")
        print(f"{schema['interpretation']}")
        
        print(f"\n{'='*100}")
        print(f"OVERALL: {validation_result['summary']}")
        print("="*100 + "\n")


# Test if run directly
if __name__ == "__main__":
    import os
    
    print("\n" + "="*80)
    print("TESTING: LLM Component Validator")
    print("="*80)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        exit(1)
    
    validator = LLMComponentValidator()
    
    test_review = "stayed here last week. room was clean but breakfast meh."
    
    validation = validator.run_full_validation(test_review, n_consistency_runs=3)
    validator.pretty_print_validation(validation)
    
    print("="*80)
    print("Component Validator working correctly!")
    print("="*80 + "\n")
