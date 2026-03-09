"""Tests for claude_toolkit.tokenizer.counter"""

import pytest
from claude_toolkit.tokenizer.counter import (
    count_tokens,
    count_sections,
    section_heatmap,
    estimate_output_tokens,
)


class TestCountTokens:
    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_whitespace_only_returns_zero(self):
        assert count_tokens("   \n\t  ") == 0

    def test_single_word(self):
        assert count_tokens("hello") >= 1

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("Hi")
        long  = count_tokens("This is a much longer sentence with many words in it.")
        assert long > short

    def test_returns_positive_integer(self):
        result = count_tokens("test prompt", "claude-sonnet")
        assert isinstance(result, int)
        assert result >= 1

    def test_model_parameter_accepted(self):
        for model in ("claude-sonnet", "claude-opus", "claude-haiku"):
            result = count_tokens("some text", model)
            assert result >= 1


class TestCountSections:
    def test_returns_dict(self):
        result = count_sections("Hello world this is a test prompt.")
        assert isinstance(result, dict)

    def test_sections_have_positive_counts(self):
        prompt = "Instructions:\nWrite a function.\n\nContext:\nPython 3.10."
        result = count_sections(prompt)
        assert all(v > 0 for v in result.values())

    def test_code_section_detected(self):
        prompt = "Here is code:\n```python\ndef foo(): pass\n```"
        result = count_sections(prompt)
        assert "code" in result


class TestSectionHeatmap:
    def test_values_sum_near_100(self):
        prompt = "Instructions:\nDo something.\n\nContext:\nBackground info here.\n"
        result = section_heatmap(prompt)
        if result:
            total = sum(result.values())
            assert abs(total - 100.0) < 1.0

    def test_returns_floats(self):
        result = section_heatmap("A normal prompt with no special headers.")
        assert all(isinstance(v, float) for v in result.values())


class TestEstimateOutputTokens:
    def test_generation_prompt_has_higher_ratio(self):
        # Use 200-word prompts so both estimates exceed the 50-token floor
        base = "word " * 200
        gen    = estimate_output_tokens("Generate and implement and write and build and create and code " + base)
        summary = estimate_output_tokens("Summarize briefly and shorten and tldr " + base)
        assert gen > summary

    def test_minimum_floor_of_50(self):
        assert estimate_output_tokens("Yes?") >= 50

    def test_returns_positive_int(self):
        result = estimate_output_tokens("Explain machine learning.")
        assert isinstance(result, int)
        assert result > 0
