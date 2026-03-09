"""
CLAUDE.md context extractor.

Analyses session prompts to identify repeated context blocks — project
architecture notes, authentication patterns, database schemas, etc. —
and generates a reusable CLAUDE.md file that replaces per-prompt injection.

Approach
--------
1. Extract individual sentences from every prompt.
2. Count how frequently each sentence (and sliding n-gram phrase) appears
   across the session using a Counter.
3. Filter to phrases that appear in >= min_occurrences prompts and exceed
   a minimum token size.
4. Deduplicate (remove phrases that are substrings of a longer phrase
   already captured).
5. Categorise each block by domain heuristics (auth, DB, API, etc.).
6. Render a formatted CLAUDE.md with a section per category.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from claude_toolkit.models import ClaudeMdResult, ContextBlock
from claude_toolkit.tokenizer.counter import count_tokens


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "project":        [r"project", r"codebase", r"repository", r"repo", r"architecture"],
    "authentication": [r"auth(?:entication)?", r"jwt", r"oauth", r"login", r"session\b"],
    "database":       [r"database", r"schema", r"table\b", r"sql\b", r"mongodb", r"postgres"],
    "api":            [r"\bapi\b", r"endpoint", r"\brest\b", r"graphql", r"webhook"],
    "stack":          [r"\bstack\b", r"framework", r"django", r"react", r"fastapi", r"flask"],
    "conventions":    [r"convention", r"style\b", r"naming", r"format\b", r"guideline"],
    "testing":        [r"\btest\b", r"pytest", r"unittest", r"coverage"],
}


def _categorize(text: str) -> str:
    lower = text.lower()
    for category, patterns in _CATEGORY_PATTERNS.items():
        for p in patterns:
            if re.search(p, lower):
                return category
    return "general"


# ---------------------------------------------------------------------------
# Phrase extraction
# ---------------------------------------------------------------------------

def _extract_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering very short ones."""
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip().split()) >= 6
    ]


def _ngram_phrases(text: str, min_words: int = 8, max_words: int = 20) -> List[str]:
    """Yield sliding window phrases of length min_words..max_words."""
    words = text.split()
    phrases: List[str] = []
    for window in range(min_words, min(max_words + 1, len(words) + 1)):
        for i in range(len(words) - window + 1):
            phrases.append(" ".join(words[i : i + window]))
    return phrases


def _find_repeated_phrases(
    prompts: List[str],
    min_occurrences: int = 2,
    min_words: int = 8,
) -> List[Tuple[str, int]]:
    """
    Return (phrase, count) pairs where the phrase appears in at least
    min_occurrences distinct prompts, sorted by phrase length descending.
    """
    phrase_counts: Counter = Counter()

    for prompt in prompts:
        # Sentence-level
        for sentence in _extract_sentences(prompt):
            key = re.sub(r"\s+", " ", sentence.lower().strip())
            phrase_counts[key] += 1

        # N-gram sliding window
        seen_in_prompt: set = set()
        for phrase in _ngram_phrases(prompt, min_words):
            key = re.sub(r"\s+", " ", phrase.lower().strip())
            if key not in seen_in_prompt:
                phrase_counts[key] += 1
                seen_in_prompt.add(key)

    repeated = [
        (phrase, count)
        for phrase, count in phrase_counts.items()
        if count >= min_occurrences
    ]
    # Sort: longer phrases first (more context), then by frequency
    repeated.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
    return repeated[:30]


# ---------------------------------------------------------------------------
# Block extraction & deduplication
# ---------------------------------------------------------------------------

def extract_context_blocks(
    prompts: List[str],
    min_occurrences: int = 2,
    min_tokens_per_block: int = 15,
) -> List[ContextBlock]:
    """
    Extract repeated context blocks from a list of prompt strings.

    Returns a deduplicated list of ContextBlock objects, sorted by token
    count descending.
    """
    repeated = _find_repeated_phrases(prompts, min_occurrences)
    blocks: List[ContextBlock] = []
    seen_keys: List[str] = []  # track added phrases to avoid substrings

    for phrase, count in repeated:
        # Skip if phrase is a substring of something already added
        if any(phrase in existing for existing in seen_keys):
            continue

        tokens = count_tokens(phrase)
        if tokens < min_tokens_per_block:
            continue

        seen_keys.append(phrase)
        blocks.append(ContextBlock(
            content=phrase.strip(),
            token_count=tokens,
            occurrences=count,
            category=_categorize(phrase),
        ))

        if len(blocks) >= 12:
            break

    blocks.sort(key=lambda b: b.token_count, reverse=True)
    return blocks


# ---------------------------------------------------------------------------
# CLAUDE.md generation
# ---------------------------------------------------------------------------

_CLAUDE_MD_HEADER = """\
# {project_name} — Shared Context (CLAUDE.md)

<!-- Auto-generated by claude-toolkit v2 context-extractor.
     Review, edit, and commit this file to your repository.
     Claude Code will automatically inject it into every session. -->

"""

_CLAUDE_MD_FOOTER = """\

---
*Auto-generated from {n_prompts} session prompts.*
*Detected {n_blocks} repeated context blocks totalling {total_tokens:,} tokens.*
*Estimated savings: ~{savings_per_prompt:,} tokens per prompt.*
"""


def generate_claude_md(
    prompts: List[str],
    project_name: str = "Project",
    output_path: Optional[str] = None,
    min_occurrences: int = 2,
) -> ClaudeMdResult:
    """
    Analyse session prompts and produce a reusable CLAUDE.md draft.

    Args:
        prompts:         List of raw prompt strings from a session.
        project_name:    Project name to use in the CLAUDE.md header.
        output_path:     If given, write the file to this path.
        min_occurrences: Minimum occurrences for a phrase to be included.

    Returns:
        ClaudeMdResult with generated markdown content and savings estimates.
    """
    blocks = extract_context_blocks(prompts, min_occurrences)

    if not blocks:
        content = _CLAUDE_MD_HEADER.format(project_name=project_name)
        content += "<!-- No repeated context detected. Add your project context here. -->\n"
        content += "\n## Project Overview\n\n- \n\n## Key Conventions\n\n- \n"
        return ClaudeMdResult(
            blocks=[],
            generated_content=content,
            total_repeated_tokens=0,
            estimated_savings_per_prompt=0,
            output_path=output_path,
        )

    # Group by category
    by_category: Dict[str, List[ContextBlock]] = {}
    for b in blocks:
        by_category.setdefault(b.category, []).append(b)

    lines: List[str] = [_CLAUDE_MD_HEADER.format(project_name=project_name).rstrip()]

    for category, cat_blocks in sorted(by_category.items()):
        lines.append(f"\n## {category.title()}\n")
        for b in cat_blocks:
            # Restore sentence-case from first word
            text = b.content.strip()
            if text and not text[0].isupper():
                text = text[0].upper() + text[1:]
            lines.append(f"- {text}")
        lines.append("")

    total_repeated = sum(b.token_count * b.occurrences for b in blocks)
    savings_per_prompt = sum(b.token_count for b in blocks)

    lines.append(
        _CLAUDE_MD_FOOTER.format(
            n_prompts=len(prompts),
            n_blocks=len(blocks),
            total_tokens=total_repeated,
            savings_per_prompt=savings_per_prompt,
        ).strip()
    )

    content = "\n".join(lines) + "\n"

    if output_path:
        Path(output_path).write_text(content, encoding="utf-8")

    return ClaudeMdResult(
        blocks=blocks,
        generated_content=content,
        total_repeated_tokens=total_repeated,
        estimated_savings_per_prompt=savings_per_prompt,
        output_path=output_path,
    )
