"""Tests for claude_toolkit.example_pruner.pruner"""

import pytest
from claude_toolkit.example_pruner.pruner import prune_examples, _greedy_cluster, _similarity_matrix
from claude_toolkit.models import PruningResult, ExampleCluster

PROMPT_WITH_EXAMPLES = """
Classify the sentiment of the following texts.

Example 1: Input: "I absolutely love this product, it's amazing!"
Output: positive

Example 2: Input: "This is fantastic, I'm really happy with it!"
Output: positive

Example 3: Input: "I hate this, it's the worst thing ever."
Output: negative

Example 4: Input: "Terrible experience, deeply disappointed."
Output: negative

Example 5: Input: "The weather is nice today."
Output: neutral
"""

PROMPT_NO_EXAMPLES = "Write a function to reverse a string in Python."

PROMPT_ONE_EXAMPLE = "For example, use list comprehensions instead of loops when possible."


class TestPruneExamples:
    def test_returns_pruning_result(self):
        result = prune_examples(PROMPT_WITH_EXAMPLES)
        assert isinstance(result, PruningResult)

    def test_original_examples_detected(self):
        result = prune_examples(PROMPT_WITH_EXAMPLES)
        assert len(result.original_examples) >= 2

    def test_kept_examples_lte_original(self):
        result = prune_examples(PROMPT_WITH_EXAMPLES)
        assert len(result.kept_examples) <= len(result.original_examples)

    def test_tokens_saved_non_negative(self):
        result = prune_examples(PROMPT_WITH_EXAMPLES)
        assert result.tokens_saved >= 0

    def test_reduction_pct_in_range(self):
        result = prune_examples(PROMPT_WITH_EXAMPLES)
        assert 0.0 <= result.reduction_pct <= 100.0

    def test_pruned_tokens_lte_original_tokens(self):
        result = prune_examples(PROMPT_WITH_EXAMPLES)
        assert result.pruned_tokens <= result.original_tokens

    def test_no_examples_returns_empty_clusters(self):
        result = prune_examples(PROMPT_NO_EXAMPLES)
        assert result.clusters == []
        assert result.tokens_saved == 0

    def test_single_example_no_pruning(self):
        result = prune_examples(PROMPT_ONE_EXAMPLE)
        assert result.tokens_saved == 0

    def test_clusters_are_example_cluster_instances(self):
        result = prune_examples(PROMPT_WITH_EXAMPLES, similarity_threshold=0.2)
        for c in result.clusters:
            assert isinstance(c, ExampleCluster)

    def test_high_threshold_less_pruning(self):
        strict = prune_examples(PROMPT_WITH_EXAMPLES, similarity_threshold=0.99)
        loose  = prune_examples(PROMPT_WITH_EXAMPLES, similarity_threshold=0.10)
        assert strict.tokens_saved <= loose.tokens_saved

    def test_representative_is_string(self):
        result = prune_examples(PROMPT_WITH_EXAMPLES, similarity_threshold=0.2)
        for c in result.clusters:
            assert isinstance(c.representative, str)

    def test_similarity_score_in_range(self):
        result = prune_examples(PROMPT_WITH_EXAMPLES, similarity_threshold=0.2)
        for c in result.clusters:
            assert 0.0 <= c.similarity_score <= 1.0


class TestSimilarityMatrix:
    def test_diagonal_is_one(self):
        texts = ["hello world", "foo bar baz", "test example"]
        matrix = _similarity_matrix(texts)
        for i in range(len(texts)):
            assert matrix[i][i] == pytest.approx(1.0)

    def test_symmetric(self):
        texts = ["the quick brown fox", "a lazy dog sleeps", "quick brown dogs"]
        matrix = _similarity_matrix(texts)
        n = len(texts)
        for i in range(n):
            for j in range(n):
                assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-9)

    def test_values_in_range(self):
        texts = ["apple pie", "orange juice", "apple cider"]
        matrix = _similarity_matrix(texts)
        for row in matrix:
            for val in row:
                assert 0.0 <= val <= 1.0 + 1e-9

    def test_identical_texts_have_score_one(self):
        texts = ["same text here", "same text here"]
        matrix = _similarity_matrix(texts)
        assert matrix[0][1] == pytest.approx(1.0)
