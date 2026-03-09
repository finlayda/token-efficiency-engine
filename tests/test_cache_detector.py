"""Tests for claude_toolkit.cache_detector.detector"""

import pytest
from claude_toolkit.cache_detector.detector import detect_cache_candidates
from claude_toolkit.models import CacheReport, CacheCandidate

SHARED_PREFIX = "You are an expert Python engineer. The project uses FastAPI and PostgreSQL."

PROMPTS_WITH_SHARED = [
    (1, SHARED_PREFIX + " Write a login endpoint."),
    (2, SHARED_PREFIX + " Write a registration endpoint."),
    (3, SHARED_PREFIX + " Add JWT token refresh logic."),
    (4, "Summarize the README file."),
]

PROMPTS_NO_SHARED = [
    (1, "What is recursion?"),
    (2, "Explain REST APIs."),
    (3, "How does a database index work?"),
]

SINGLE_PROMPT = [(1, "Hello world.")]


class TestDetectCacheCandidates:
    def test_returns_cache_report(self):
        result = detect_cache_candidates(PROMPTS_WITH_SHARED)
        assert isinstance(result, CacheReport)

    def test_finds_shared_prefix(self):
        result = detect_cache_candidates(PROMPTS_WITH_SHARED, min_prefix_tokens=5)
        assert len(result.candidates) >= 1

    def test_candidate_frequency_correct(self):
        result = detect_cache_candidates(PROMPTS_WITH_SHARED, min_prefix_tokens=5)
        assert any(c.frequency >= 3 for c in result.candidates)

    def test_savings_pct_in_range(self):
        result = detect_cache_candidates(PROMPTS_WITH_SHARED, min_prefix_tokens=5)
        for c in result.candidates:
            assert 0.0 <= c.savings_pct <= 100.0

    def test_no_candidates_when_no_overlap(self):
        result = detect_cache_candidates(PROMPTS_NO_SHARED)
        assert result.overall_savings_pct == 0.0 or len(result.candidates) == 0

    def test_empty_input(self):
        result = detect_cache_candidates([])
        assert result.total_prompts == 0
        assert result.candidates == []

    def test_single_prompt(self):
        result = detect_cache_candidates(SINGLE_PROMPT)
        assert result.candidates == []

    def test_total_prompts_correct(self):
        result = detect_cache_candidates(PROMPTS_WITH_SHARED)
        assert result.total_prompts == len(PROMPTS_WITH_SHARED)

    def test_suggestion_is_string(self):
        result = detect_cache_candidates(PROMPTS_WITH_SHARED)
        assert isinstance(result.suggestion, str)
        assert len(result.suggestion) > 10

    def test_candidates_sorted_by_savings(self):
        result = detect_cache_candidates(PROMPTS_WITH_SHARED, min_prefix_tokens=5)
        if len(result.candidates) > 1:
            savings = [c.estimated_tokens_saved for c in result.candidates]
            assert savings == sorted(savings, reverse=True)

    def test_estimated_tokens_saved_non_negative(self):
        result = detect_cache_candidates(PROMPTS_WITH_SHARED, min_prefix_tokens=5)
        for c in result.candidates:
            assert c.estimated_tokens_saved >= 0

    def test_min_prefix_tokens_filters_short_prefixes(self):
        result_strict = detect_cache_candidates(PROMPTS_WITH_SHARED, min_prefix_tokens=1000)
        result_loose  = detect_cache_candidates(PROMPTS_WITH_SHARED, min_prefix_tokens=1)
        assert len(result_strict.candidates) <= len(result_loose.candidates)

    def test_prefix_ids_are_subset_of_input_ids(self):
        result = detect_cache_candidates(PROMPTS_WITH_SHARED, min_prefix_tokens=5)
        input_ids = {pid for pid, _ in PROMPTS_WITH_SHARED}
        for c in result.candidates:
            assert set(c.prompt_ids).issubset(input_ids)
