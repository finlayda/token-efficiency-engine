"""Tests for claude_toolkit.session_analyzer.analyzer"""

import json
import os
import tempfile
import pytest
from claude_toolkit.session_analyzer.analyzer import (
    analyze_session_data,
    analyze_session_file,
    parse_session_entries,
)
from claude_toolkit.models import SessionSummary


SAMPLE_LOG_LIST = [
    {"prompt": "What is Python?",  "response": "Python is a language.", "model": "claude-sonnet"},
    {"prompt": "Explain classes.", "response": "Classes are blueprints.", "model": "claude-sonnet"},
    {"input":  "Write a loop.",    "output":   "for i in range(10): pass", "model": "claude-haiku"},
]

SAMPLE_LOG_DICT = {
    "messages": [
        {"prompt": "Hello", "response": "Hi there!", "model": "claude-sonnet"},
        {"prompt": "Bye",   "response": "Goodbye!",  "model": "claude-sonnet"},
    ]
}

SAMPLE_WITH_TOKENS = [
    {"prompt": "Test", "response": "OK", "model": "claude-sonnet",
     "input_tokens": 10, "output_tokens": 5},
]


class TestAnalyzeSessionData:
    def test_returns_session_summary(self):
        result = analyze_session_data(SAMPLE_LOG_LIST)
        assert isinstance(result, SessionSummary)

    def test_correct_prompt_count(self):
        result = analyze_session_data(SAMPLE_LOG_LIST)
        assert result.total_prompts == 3

    def test_total_tokens_positive(self):
        result = analyze_session_data(SAMPLE_LOG_LIST)
        assert result.total_tokens > 0

    def test_total_cost_positive(self):
        result = analyze_session_data(SAMPLE_LOG_LIST)
        assert result.total_cost > 0

    def test_top_consumers_sorted_descending(self):
        result = analyze_session_data(SAMPLE_LOG_LIST)
        tokens = [e["total_tokens"] for e in result.top_consumers]
        assert tokens == sorted(tokens, reverse=True)

    def test_model_breakdown_populated(self):
        result = analyze_session_data(SAMPLE_LOG_LIST)
        assert len(result.model_breakdown) >= 1

    def test_accepts_dict_wrapper(self):
        result = analyze_session_data(SAMPLE_LOG_DICT)
        assert result.total_prompts == 2

    def test_provided_token_counts_used(self):
        result = analyze_session_data(SAMPLE_WITH_TOKENS)
        assert result.total_input_tokens == 10
        assert result.total_output_tokens == 5

    def test_empty_list_returns_zero_summary(self):
        result = analyze_session_data([])
        assert result.total_prompts == 0
        assert result.total_cost == 0.0

    def test_invalid_entries_skipped(self):
        data = [None, 42, "string", {"prompt": "Valid", "response": "Yes"}]
        result = analyze_session_data(data)
        assert result.total_prompts == 1

    def test_total_equals_input_plus_output(self):
        result = analyze_session_data(SAMPLE_LOG_LIST)
        assert result.total_tokens == result.total_input_tokens + result.total_output_tokens


class TestAnalyzeSessionFile:
    def test_reads_json_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False, encoding="utf-8") as f:
            json.dump(SAMPLE_LOG_LIST, f)
            fname = f.name
        try:
            result = analyze_session_file(fname)
            assert result.total_prompts == 3
        finally:
            os.unlink(fname)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            analyze_session_file("/nonexistent/path/session.json")

    def test_example_log_file(self):
        log_path = os.path.join(
            os.path.dirname(__file__), "..", "examples", "logs", "claude_session.json"
        )
        result = analyze_session_file(log_path)
        assert result.total_prompts > 0
        assert result.total_cost >= 0


class TestParseSessionEntries:
    def test_returns_list(self):
        entries = parse_session_entries(SAMPLE_LOG_LIST)
        assert isinstance(entries, list)

    def test_correct_count(self):
        entries = parse_session_entries(SAMPLE_LOG_LIST)
        assert len(entries) == 3

    def test_entries_have_prompt_text(self):
        entries = parse_session_entries(SAMPLE_LOG_LIST)
        for e in entries:
            assert isinstance(e.prompt, str)

    def test_dict_wrapper_parsed(self):
        entries = parse_session_entries(SAMPLE_LOG_DICT)
        assert len(entries) == 2
