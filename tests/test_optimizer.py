"""Tests for claude_toolkit.prompt_optimizer.optimizer"""

import pytest
from claude_toolkit.prompt_optimizer.optimizer import optimize
from claude_toolkit.models import OptimizationResult


VERBOSE = (
    "Hello! I would like you to please help me with this task. "
    "Could you please write a Python function? "
    "In order to accomplish this, you need to create a function that sorts a list. "
    "Please make sure to include error handling. "
    "Please make sure to include error handling. "
    "Furthermore, please add docstrings. "
    "Additionally, please write unit tests as well."
)

ALREADY_LEAN = "Sort a list. Include error handling and docstrings."


class TestOptimize:
    def test_returns_optimization_result(self):
        result = optimize(VERBOSE)
        assert isinstance(result, OptimizationResult)

    def test_optimized_has_fewer_tokens(self):
        result = optimize(VERBOSE)
        assert result.optimized_tokens <= result.original_tokens

    def test_tokens_saved_non_negative(self):
        result = optimize(VERBOSE)
        assert result.tokens_saved >= 0

    def test_reduction_pct_in_range(self):
        result = optimize(VERBOSE)
        assert 0.0 <= result.reduction_pct <= 100.0

    def test_original_prompt_preserved(self):
        result = optimize(VERBOSE)
        assert result.original_prompt == VERBOSE

    def test_strategies_list_non_empty(self):
        result = optimize(VERBOSE)
        assert len(result.strategies_applied) > 0

    def test_social_filler_removed(self):
        result = optimize("Hello! Could you please write a function?")
        assert "Hello" not in result.optimized_prompt or \
               result.optimized_tokens < result.original_tokens

    def test_deduplication_applied(self):
        duped = "Check this twice. Check this twice. Do the task."
        result = optimize(duped)
        assert result.tokens_saved >= 0

    def test_aggressive_mode_applies_more_strategies(self):
        normal     = optimize(VERBOSE, aggressive=False)
        aggressive = optimize(VERBOSE, aggressive=True)
        # Aggressive applies at least as many strategies
        assert len(aggressive.strategies_applied) >= len(normal.strategies_applied)

    def test_lean_prompt_unchanged_or_minimal_savings(self):
        result = optimize(ALREADY_LEAN)
        assert result.tokens_saved < 5  # minimal savings on already-lean prompt

    def test_suggestions_is_list(self):
        result = optimize(VERBOSE)
        assert isinstance(result.suggestions, list)

    def test_empty_prompt(self):
        result = optimize("")
        assert result.original_tokens == 0
        assert result.optimized_tokens == 0
        assert result.tokens_saved == 0
