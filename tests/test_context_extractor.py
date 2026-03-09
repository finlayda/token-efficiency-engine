"""Tests for claude_toolkit.context_extractor.extractor"""

import os
import tempfile
import pytest
from claude_toolkit.context_extractor.extractor import (
    generate_claude_md,
    extract_context_blocks,
)
from claude_toolkit.models import ClaudeMdResult, ContextBlock


REPEATED_PROMPTS = [
    "The project uses FastAPI and PostgreSQL. The auth system uses JWT tokens. Write a login endpoint.",
    "The project uses FastAPI and PostgreSQL. The auth system uses JWT tokens. Add user registration.",
    "The project uses FastAPI and PostgreSQL. The auth system uses JWT tokens. Implement logout.",
    "What is the capital of France?",
]

UNIQUE_PROMPTS = [
    "Write a quicksort algorithm.",
    "Explain binary search trees.",
    "What is Big O notation?",
]


class TestExtractContextBlocks:
    def test_returns_list(self):
        result = extract_context_blocks(REPEATED_PROMPTS)
        assert isinstance(result, list)

    def test_blocks_are_context_block_instances(self):
        result = extract_context_blocks(REPEATED_PROMPTS)
        for b in result:
            assert isinstance(b, ContextBlock)

    def test_blocks_have_positive_token_count(self):
        result = extract_context_blocks(REPEATED_PROMPTS, min_tokens_per_block=1)
        for b in result:
            assert b.token_count > 0

    def test_blocks_have_occurrences(self):
        result = extract_context_blocks(REPEATED_PROMPTS, min_occurrences=2)
        for b in result:
            assert b.occurrences >= 2

    def test_blocks_have_category(self):
        result = extract_context_blocks(REPEATED_PROMPTS, min_tokens_per_block=1)
        valid_categories = {
            "project", "authentication", "database", "api",
            "stack", "conventions", "testing", "general",
        }
        for b in result:
            assert b.category in valid_categories

    def test_no_blocks_for_unique_prompts(self):
        result = extract_context_blocks(UNIQUE_PROMPTS, min_occurrences=3)
        assert len(result) == 0

    def test_min_tokens_filter_works(self):
        result_high = extract_context_blocks(REPEATED_PROMPTS, min_tokens_per_block=1000)
        result_low  = extract_context_blocks(REPEATED_PROMPTS, min_tokens_per_block=1)
        assert len(result_high) <= len(result_low)


class TestGenerateClaudeMd:
    def test_returns_claude_md_result(self):
        result = generate_claude_md(REPEATED_PROMPTS)
        assert isinstance(result, ClaudeMdResult)

    def test_generated_content_is_string(self):
        result = generate_claude_md(REPEATED_PROMPTS)
        assert isinstance(result.generated_content, str)
        assert len(result.generated_content) > 0

    def test_project_name_in_header(self):
        result = generate_claude_md(REPEATED_PROMPTS, project_name="TestProject")
        assert "TestProject" in result.generated_content

    def test_savings_per_prompt_non_negative(self):
        result = generate_claude_md(REPEATED_PROMPTS)
        assert result.estimated_savings_per_prompt >= 0

    def test_total_repeated_tokens_non_negative(self):
        result = generate_claude_md(REPEATED_PROMPTS)
        assert result.total_repeated_tokens >= 0

    def test_unique_prompts_produce_empty_template(self):
        result = generate_claude_md(UNIQUE_PROMPTS, min_occurrences=5)
        assert result.blocks == []
        assert "Add your project context" in result.generated_content

    def test_writes_file_when_output_path_given(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            result = generate_claude_md(REPEATED_PROMPTS, output_path=path)
            assert os.path.exists(path)
            content = open(path, encoding="utf-8").read()
            assert len(content) > 0
        finally:
            os.unlink(path)

    def test_no_file_written_when_path_is_none(self):
        # Should not raise and output_path remains None
        result = generate_claude_md(REPEATED_PROMPTS, output_path=None)
        assert result.output_path is None

    def test_content_contains_markdown_headers(self):
        result = generate_claude_md(REPEATED_PROMPTS)
        assert "#" in result.generated_content
