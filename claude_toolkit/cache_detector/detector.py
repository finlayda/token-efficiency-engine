"""
Prompt prefix cache detection.

Detects repeated prefixes and shared system prompts across a session
and quantifies the token savings achievable via Anthropic prompt caching.

Algorithm
---------
1. Exact common-prefix scan: find the longest word-aligned prefix shared
   by the maximum number of prompts.
2. Similarity clustering: use SequenceMatcher to find prompts that share
   a substantial common opening block even when they diverge mid-way.
3. Deduplicate candidates by content and rank by estimated savings.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from claude_toolkit.models import CacheCandidate, CacheReport
from claude_toolkit.tokenizer.counter import count_tokens


# ---------------------------------------------------------------------------
# Prefix detection helpers
# ---------------------------------------------------------------------------

def _longest_common_prefix(a: str, b: str) -> str:
    """Return the longest word-aligned common prefix between two strings."""
    words_a = a.split()
    words_b = b.split()
    n = 0
    for wa, wb in zip(words_a, words_b):
        if wa == wb:
            n += 1
        else:
            break
    return " ".join(words_a[:n])


def _find_exact_prefix_candidate(
    prompts: List[Tuple[int, str]],
    min_tokens: int = 50,
) -> Optional[CacheCandidate]:
    """
    Find the longest prefix shared between the first two prompts, then count
    how many total prompts also start with that prefix.
    """
    if len(prompts) < 2:
        return None

    best_prefix = _longest_common_prefix(prompts[0][1], prompts[1][1])
    if not best_prefix:
        return None

    matching_ids: List[int] = []
    for pid, text in prompts:
        if text.startswith(best_prefix):
            matching_ids.append(pid)

    if len(matching_ids) < 2:
        return None

    prefix_tokens = count_tokens(best_prefix)
    if prefix_tokens < min_tokens:
        return None

    total_input = sum(count_tokens(t) for _, t in prompts)
    saved = prefix_tokens * (len(matching_ids) - 1)
    savings_pct = round(saved / total_input * 100, 1) if total_input > 0 else 0.0

    return CacheCandidate(
        prefix=best_prefix,
        token_count=prefix_tokens,
        frequency=len(matching_ids),
        prompt_ids=matching_ids,
        savings_pct=savings_pct,
        estimated_tokens_saved=saved,
    )


def _find_similarity_clusters(
    prompts: List[Tuple[int, str]],
    similarity_threshold: float = 0.6,
    min_tokens: int = 30,
) -> List[CacheCandidate]:
    """
    Greedy similarity clustering: group prompts whose first 500 characters
    are substantially similar (SequenceMatcher ratio >= threshold), then
    extract the common prefix within each group.
    """
    candidates: List[CacheCandidate] = []
    processed: set = set()
    total_input = sum(count_tokens(t) for _, t in prompts)

    for i, (pid_a, text_a) in enumerate(prompts):
        if i in processed:
            continue

        group_ids = [pid_a]
        group_texts = [text_a]

        for j, (pid_b, text_b) in enumerate(prompts[i + 1:], i + 1):
            if j in processed:
                continue
            ratio = SequenceMatcher(None, text_a[:500], text_b[:500]).ratio()
            if ratio >= similarity_threshold:
                group_ids.append(pid_b)
                group_texts.append(text_b)
                processed.add(j)

        if len(group_ids) < 2:
            continue

        # Reduce to common prefix across the entire group
        common = group_texts[0]
        for t in group_texts[1:]:
            common = _longest_common_prefix(common, t)
            if not common:
                break

        if not common:
            continue

        prefix_tokens = count_tokens(common)
        if prefix_tokens < min_tokens:
            continue

        saved = prefix_tokens * (len(group_ids) - 1)
        savings_pct = round(saved / total_input * 100, 1) if total_input > 0 else 0.0
        display_prefix = common[:300] + ("..." if len(common) > 300 else "")

        candidates.append(CacheCandidate(
            prefix=display_prefix,
            token_count=prefix_tokens,
            frequency=len(group_ids),
            prompt_ids=group_ids,
            savings_pct=savings_pct,
            estimated_tokens_saved=saved,
        ))
        processed.add(i)

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_cache_candidates(
    prompts: List[Tuple[int, str]],
    min_prefix_tokens: int = 50,
    similarity_threshold: float = 0.6,
) -> CacheReport:
    """
    Analyse (prompt_id, prompt_text) pairs for prompt-caching opportunities.

    Args:
        prompts:              List of (id, text) tuples from a session.
        min_prefix_tokens:    Minimum shared prefix size to flag as cacheable.
        similarity_threshold: SequenceMatcher ratio for grouping similar prompts.

    Returns:
        CacheReport with candidates ranked by estimated token savings.
    """
    if not prompts:
        return CacheReport(
            candidates=[],
            total_prompts=0,
            total_cacheable_tokens=0,
            overall_savings_pct=0.0,
            suggestion="No prompts provided.",
        )

    candidates: List[CacheCandidate] = []

    exact = _find_exact_prefix_candidate(prompts, min_prefix_tokens)
    if exact:
        candidates.append(exact)

    similar = _find_similarity_clusters(prompts, similarity_threshold, min_prefix_tokens)
    candidates.extend(similar)

    # Deduplicate by prefix content (avoid near-identical candidates)
    seen: set = set()
    unique: List[CacheCandidate] = []
    for c in sorted(candidates, key=lambda x: x.estimated_tokens_saved, reverse=True):
        key = c.prefix[:80].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    total_input = sum(count_tokens(t) for _, t in prompts)
    total_cacheable = sum(c.estimated_tokens_saved for c in unique)
    overall_pct = round(total_cacheable / total_input * 100, 1) if total_input > 0 else 0.0

    if unique:
        top = unique[0]
        suggestion = (
            f"Move the {top.token_count:,}-token shared prefix (used in {top.frequency} prompts) "
            "to a stable system prompt and enable Anthropic prompt caching "
            "(cache_control: ephemeral) to eliminate redundant token charges on each call. "
            f"Estimated session savings: {overall_pct:.0f}%."
        )
    else:
        suggestion = (
            "No strong caching candidates found. Ensure system prompts are stable "
            "across turns — identical prefixes are automatically cached by Anthropic "
            "when prompt caching is enabled."
        )

    return CacheReport(
        candidates=unique,
        total_prompts=len(prompts),
        total_cacheable_tokens=total_cacheable,
        overall_savings_pct=overall_pct,
        suggestion=suggestion,
    )
