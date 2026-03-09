"""Tests for claude_toolkit.heatmap.heatmap"""

import pytest
from claude_toolkit.heatmap.heatmap import render_heatmap_text, get_token_density


MULTI_SECTION_PROMPT = """\
Instructions:
Write a function that reverses a list.

Context:
Python 3.10. Standard library only.

Examples:
Input: [1, 2, 3] -> Output: [3, 2, 1]
"""


class TestRenderHeatmapText:
    def test_returns_string(self):
        result = render_heatmap_text(MULTI_SECTION_PROMPT)
        assert isinstance(result, str)

    def test_non_empty_for_real_prompt(self):
        result = render_heatmap_text(MULTI_SECTION_PROMPT)
        assert len(result.strip()) > 0

    def test_empty_prompt_returns_no_sections(self):
        result = render_heatmap_text("")
        assert "no sections" in result.lower() or result.strip() == ""

    def test_contains_percentage(self):
        result = render_heatmap_text(MULTI_SECTION_PROMPT)
        assert "%" in result


class TestGetTokenDensity:
    def test_returns_list_of_tuples(self):
        result = get_token_density(MULTI_SECTION_PROMPT)
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 3 for t in result)

    def test_line_count_matches(self):
        n_lines = len(MULTI_SECTION_PROMPT.split("\n"))
        result  = get_token_density(MULTI_SECTION_PROMPT)
        assert len(result) == n_lines

    def test_empty_lines_have_zero_tokens(self):
        result = get_token_density(MULTI_SECTION_PROMPT)
        for line_text, tokens, label in result:
            if not line_text.strip():
                assert tokens == 0

    def test_non_empty_lines_have_positive_tokens(self):
        result = get_token_density("Hello world this is a test.")
        assert any(t > 0 for _, t, _ in result)

    def test_label_is_string(self):
        result = get_token_density("Some text here.")
        for _, _, label in result:
            assert isinstance(label, str)
