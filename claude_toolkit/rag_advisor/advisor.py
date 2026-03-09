"""
RAG (Retrieval-Augmented Generation) context compression advisor.

Detects large static context blocks embedded inside prompts — pasted
documents, code files, data tables — and recommends replacing them with
retrieval-based injection so only the relevant excerpt is included at
query time.

Detection strategy
------------------
1. Scan the prompt line-by-line, accumulating tokens until a contiguous
   block exceeds the threshold.
2. Classify each block as: code, document, data, or freetext.
3. Map each block type to a retrieval suggestion.
4. Estimate a 70% savings rate (empirical average for chunked RAG pipelines).
"""

import re
from typing import List, Tuple

from claude_toolkit.models import RagAdvice, RagContextBlock
from claude_toolkit.tokenizer.counter import count_tokens


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LARGE_BLOCK_THRESHOLD = 300        # tokens — tunable via analyze_rag_opportunities()

_RETRIEVAL_SUGGESTIONS: dict = {
    "code": (
        "Reference this code by file path (@src/file.py in Claude Code) rather than "
        "pasting it inline. For larger codebases use a code-search tool to retrieve "
        "only the relevant function or class."
    ),
    "document": (
        "Embed this document in a vector store (e.g., Chroma, Pinecone, pgvector) "
        "and retrieve only the top-k semantically relevant chunks at query time."
    ),
    "data": (
        "Load this data into a structured store (SQLite, DuckDB, CSV) and inject "
        "only the rows/columns relevant to the current query via a SQL or filter call."
    ),
    "freetext": (
        "Break this block into smaller chunks, embed them, and retrieve only the "
        "most relevant passage. A 512-token chunk size is a good starting point."
    ),
}


# ---------------------------------------------------------------------------
# Block detection helpers
# ---------------------------------------------------------------------------

def _classify_block(content: str) -> str:
    """Heuristically classify a text block as code / document / data / freetext."""
    if re.search(r"```", content):
        return "code"
    if re.search(r"(?m)^(?:\d+\.\s+|[-*•]\s+)", content) and content.count("\n") > 3:
        return "data"
    if re.search(r"(?m)^#{1,3}\s+", content):
        return "document"
    return "freetext"


def _find_large_blocks(
    prompt: str,
    threshold: int,
) -> List[Tuple[int, int, str]]:
    """
    Scan prompt lines and return (start_line, end_line, content) for each
    contiguous block whose token count reaches *threshold*.

    Uses a greedy accumulator: once a block crosses the threshold it is
    emitted and the accumulator resets.
    """
    lines = prompt.split("\n")
    results: List[Tuple[int, int, str]] = []
    buffer: List[str] = []
    start = 0

    for i, line in enumerate(lines):
        buffer.append(line)
        if count_tokens("\n".join(buffer)) >= threshold:
            results.append((start, i, "\n".join(buffer)))
            start = i + 1
            buffer = []

    # Residual block
    if buffer:
        residual_tokens = count_tokens("\n".join(buffer))
        if residual_tokens >= threshold:
            results.append((start, len(lines) - 1, "\n".join(buffer)))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_rag_opportunities(
    prompt: str,
    threshold_tokens: int = _LARGE_BLOCK_THRESHOLD,
) -> RagAdvice:
    """
    Analyse *prompt* for large static context blocks that could be replaced
    by retrieval-based injection.

    Args:
        prompt:           The prompt text to analyse.
        threshold_tokens: Minimum token count to flag a block (default 300).

    Returns:
        RagAdvice with detected blocks, savings estimates, and recommendations.
    """
    total_tokens = count_tokens(prompt)
    raw_blocks = _find_large_blocks(prompt, threshold_tokens)
    context_blocks: List[RagContextBlock] = []
    total_context_tokens = 0

    for start, end, content in raw_blocks:
        block_tokens = count_tokens(content)
        block_type = _classify_block(content)
        # ~70 % is a conservative estimate for well-tuned RAG pipelines
        savings_pct = 70.0

        context_blocks.append(RagContextBlock(
            start_line=start,
            end_line=end,
            content=content[:200] + ("..." if len(content) > 200 else ""),
            token_count=block_tokens,
            block_type=block_type,
            retrieval_suggestion=_RETRIEVAL_SUGGESTIONS[block_type],
            estimated_savings_pct=savings_pct,
        ))
        total_context_tokens += block_tokens

    context_fraction = (
        round(total_context_tokens / total_tokens * 100, 1) if total_tokens > 0 else 0.0
    )
    total_savings_pct = (
        round(
            sum(b.token_count * b.estimated_savings_pct / 100 for b in context_blocks)
            / total_tokens * 100,
            1,
        )
        if total_tokens > 0
        else 0.0
    )

    recommendations: List[str] = []
    if not context_blocks:
        recommendations.append(
            "No large static context blocks detected — prompt is well-scoped."
        )
    else:
        recommendations.append(
            f"Found {len(context_blocks)} large context block(s) totalling "
            f"{total_context_tokens:,} tokens ({context_fraction:.0f}% of prompt). "
            f"Switching to retrieval could shrink this prompt by ~{total_savings_pct:.0f}%."
        )
        recommendations.append(
            "Recommended pipeline: "
            "chunk -> embed (text-embedding-3-small or voyage-3) -> "
            "store -> retrieve top-k at query time."
        )
        if any(b.block_type == "code" for b in context_blocks):
            recommendations.append(
                "For code context, use Claude Code's @file reference syntax "
                "to avoid pasting entire files."
            )
        if total_tokens > 4_000:
            projected = int(total_tokens * (1 - total_savings_pct / 100))
            recommendations.append(
                f"Estimated post-RAG prompt size: "
                f"{total_tokens:,} → ~{projected:,} tokens."
            )

    return RagAdvice(
        blocks=context_blocks,
        total_context_tokens=total_context_tokens,
        total_prompt_tokens=total_tokens,
        context_fraction=context_fraction,
        total_estimated_savings_pct=total_savings_pct,
        recommendations=recommendations,
    )
