"""Tests for claude_toolkit.rag_advisor.advisor"""

import pytest
from claude_toolkit.rag_advisor.advisor import analyze_rag_opportunities
from claude_toolkit.models import RagAdvice, RagContextBlock

SHORT_PROMPT = "Summarize this article in one sentence."

LARGE_CONTEXT_PROMPT = (
    "Analyze the following document and extract key insights.\n\n"
    + ("This is a very important sentence that contains meaningful information. " * 60)
    + "\n\nProvide a bullet-point summary."
)

CODE_PROMPT = (
    "Review this code:\n```python\n"
    + "\n".join(f"def function_{i}(): return {i}" for i in range(40))
    + "\n```\nWhat improvements would you suggest?"
)


class TestAnalyzeRagOpportunities:
    def test_returns_rag_advice(self):
        result = analyze_rag_opportunities(SHORT_PROMPT)
        assert isinstance(result, RagAdvice)

    def test_no_blocks_for_short_prompt(self):
        result = analyze_rag_opportunities(SHORT_PROMPT, threshold_tokens=300)
        assert len(result.blocks) == 0

    def test_detects_large_context_block(self):
        result = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT, threshold_tokens=100)
        assert len(result.blocks) >= 1

    def test_blocks_are_rag_context_block_instances(self):
        result = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT, threshold_tokens=100)
        for b in result.blocks:
            assert isinstance(b, RagContextBlock)

    def test_block_type_is_valid(self):
        result = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT, threshold_tokens=100)
        valid_types = {"code", "document", "data", "freetext"}
        for b in result.blocks:
            assert b.block_type in valid_types

    def test_code_block_classified_as_code(self):
        result = analyze_rag_opportunities(CODE_PROMPT, threshold_tokens=50)
        types = {b.block_type for b in result.blocks}
        assert "code" in types

    def test_savings_pct_in_range(self):
        result = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT, threshold_tokens=100)
        for b in result.blocks:
            assert 0.0 <= b.estimated_savings_pct <= 100.0

    def test_total_context_tokens_non_negative(self):
        result = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT)
        assert result.total_context_tokens >= 0

    def test_context_fraction_in_range(self):
        result = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT, threshold_tokens=100)
        assert 0.0 <= result.context_fraction <= 100.0

    def test_recommendations_non_empty(self):
        result = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT, threshold_tokens=100)
        assert len(result.recommendations) >= 1

    def test_recommendations_are_strings(self):
        result = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT, threshold_tokens=100)
        for r in result.recommendations:
            assert isinstance(r, str)

    def test_total_prompt_tokens_positive(self):
        result = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT)
        assert result.total_prompt_tokens > 0

    def test_threshold_controls_sensitivity(self):
        strict = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT, threshold_tokens=5000)
        loose  = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT, threshold_tokens=50)
        assert len(strict.blocks) <= len(loose.blocks)

    def test_retrieval_suggestion_non_empty(self):
        result = analyze_rag_opportunities(LARGE_CONTEXT_PROMPT, threshold_tokens=100)
        for b in result.blocks:
            assert isinstance(b.retrieval_suggestion, str)
            assert len(b.retrieval_suggestion) > 10
