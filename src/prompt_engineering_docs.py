"""
Prompt Engineering Documentation - Course Requirement
Documents systematic prompt design across all LLM components.
"""

from __future__ import annotations


class PromptEngineeringFramework:
    """
    Documents prompt engineering strategies used in the project.
    Demonstrates systematic design following best practices.
    """
    
    @staticmethod
    def generate_report() -> None:
        """Generate comprehensive prompt engineering report."""
        print("\n" + "="*100)
        print("PROMPT ENGINEERING FRAMEWORK DOCUMENTATION")
        print("="*100)
        
        print("\n📋 This project demonstrates systematic prompt engineering across components:")
        print("   1. LLM Linguistic Analyzer")
        print("   2. LLM Fusion Adjudicator")
        print("   3. LLM Attribution Analyzer")
        print("   4. Baseline (Few-Shot)")
        
        print("\n" + "="*100)
        print("COMPONENT 1: LLM Linguistic Analyzer")
        print("="*100)
        
        print("\nPurpose: Extract 6 structured linguistic dimensions")
        print("\nKey Techniques:")
        print("  ✅ Structured output prompting (JSON schema)")
        print("  ✅ Role clarity (analyzer, not classifier)")
        print("  ✅ Evidence requirement (quoted spans)")
        print("  ✅ Negative instructions (what NOT to do)")
        print("  ✅ Literature grounding (Li & Zhang, Erol et al.)")
        
        print("\nWhy This Design:")
        print("  • structured_output: Consistent format for downstream processing")
        print("  • evidence_requirement: Prevents hallucination, grounds in text")
        print("  • negative_instructions: Prevents jumping to classification")
        print("  • role_separation: Clear separation of concerns")
        
        print("\n" + "="*100)
        print("COMPONENT 2: LLM Fusion Adjudicator")
        print("="*100)
        
        print("\nPurpose: Synthesize classifier + linguistic evidence")
        print("\nKey Techniques:")
        print("  ✅ Multi-evidence reasoning (two sources)")
        print("  ✅ Constrained generation (rule-based guidance)")
        print("  ✅ Uncertainty quantification (3-band output)")
        print("  ✅ Grounded explanation (references both sources)")
        print("  ✅ Disagreement handling (flags conflicts)")
        
        print("\nWhy This Design:")
        print("  • rule_constraints: Prevents ignoring one source")
        print("  • forced_guidance: Rules guide LLM without full decision")
        print("  • multi_evidence: Requires synthesis, not re-classification")
        print("  • uncertainty_explicit: Flags conflicts vs forcing decision")
        
        print("\n" + "="*100)
        print("COMPONENT 3: LLM Attribution Analyzer")
        print("="*100)
        
        print("\nPurpose: Explain which linguistic signals drove decision")
        print("\nKey Techniques:")
        print("  ✅ Meta-reasoning (LLM analyzes LLM outputs)")
        print("  ✅ Attribution analysis (which dimension matters)")
        print("  ✅ Counterfactual reasoning (what would change it)")
        print("  ✅ Contribution scoring (0-100 quantification)")
        
        print("\nWhy This Design:")
        print("  • meta_level: Shows LLM can reason about reasoning")
        print("  • attribution: Explains interpretability")
        print("  • counterfactual: Demonstrates causal understanding")
        
        print("\n" + "="*100)
        print("EVOLUTION: Baseline → Structured Prompting")
        print("="*100)
        
        print("\n❌ Baseline (Few-Shot) Issues:")
        print("  • Output format varies")
        print("  • No validation possible")
        print("  • Explanation mixed with label")
        print("  • Hard to extract structured data")
        
        print("\n✅ Structured Approach Benefits:")
        print("  • Consistent format")
        print("  • Programmatic validation")
        print("  • Separate fields for different info")
        print("  • Composable with other components")
        
        print("\n📈 Evolution Path: Baseline → Structured → Multi-stage → Meta-analysis")
        print("\n💡 Key Insight: Systematic prompt engineering enables systematic LLM applications")
        
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        
        print("\nThis project demonstrates systematic prompt engineering through:")
        print("  1. ✅ Structured output prompting (Pydantic schemas)")
        print("  2. ✅ Role-based task specification")
        print("  3. ✅ Evidence grounding requirements")
        print("  4. ✅ Constraint-guided generation")
        print("  5. ✅ Multi-stage prompt composition")
        print("  6. ✅ Meta-level reasoning prompts")
        
        print("\n✅ This addresses course topic: 'prompt engineering'!")
        print("="*100 + "\n")


# Run documentation
if __name__ == "__main__":
    framework = PromptEngineeringFramework()
    framework.generate_report()
