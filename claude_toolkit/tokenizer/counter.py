"""
Token counting and section analysis for Claude prompts.

Uses tiktoken (cl100k_base) as a close approximation for Claude's tokenizer.
Falls back to a word/character heuristic if tiktoken is unavailable.
"""

import re
from typing import Dict, Tuple

try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _encoder = None
    _TIKTOKEN_AVAILABLE = False

# Claude's tokenizer is similar to cl100k_base but slightly more efficient
# on prose. A 0.95 correction factor is a conservative approximation.
_CLAUDE_CORRECTION = 0.95


def count_tokens(text: str, model: str = "claude-sonnet") -> int:
    """Count approximate tokens in text.

    Uses tiktoken cl100k_base with a small correction factor when available;
    falls back to a word+character blend heuristic otherwise.
    """
    if not text or not text.strip():
        return 0

    if _TIKTOKEN_AVAILABLE:
        raw = len(_encoder.encode(text))
        return max(1, int(raw * _CLAUDE_CORRECTION))

    # Heuristic fallback: blend word count and character count estimates.
    # ~1.3 tokens/word and ~4 chars/token are well-established approximations.
    words = len(text.split())
    chars = len(text)
    word_estimate = int(words * 1.3)
    char_estimate = int(chars / 4)
    return max(1, (word_estimate + char_estimate) // 2)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

_SECTION_PATTERNS: Dict[str, str] = {
    "system":       r"(?:^|\n)(?:system|SYSTEM)\s*:?\s*\n(.*?)(?=\n(?:user|human|assistant|USER|HUMAN|ASSISTANT)\s*:|$)",
    "context":      r"(?:context|CONTEXT|background|BACKGROUND)\s*:?\s*\n(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",
    "instructions": r"(?:instructions?|INSTRUCTIONS?|task|TASK|objective|OBJECTIVE)\s*:?\s*\n(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",
    "examples":     r"(?:examples?|EXAMPLES?|sample|SAMPLE)\s*:?\s*\n(.*?)(?=\n[A-Z][A-Za-z ]+:|$)",
    "code":         r"```[\w]*\n(.*?)```",
}


def _detect_sections(prompt: str) -> Dict[str, str]:
    """Split a prompt into named logical sections."""
    sections: Dict[str, str] = {}

    for name, pattern in _SECTION_PATTERNS.items():
        matches = re.findall(pattern, prompt, re.DOTALL | re.IGNORECASE)
        if matches:
            sections[name] = " ".join(m.strip() for m in matches)

    # Fallback: divide by line thirds when no headers are found
    if not sections:
        lines = prompt.split("\n")
        n = len(lines)
        if n <= 3:
            sections["instruction"] = prompt
        else:
            third = max(1, n // 3)
            sections["preamble"]     = "\n".join(lines[:third])
            sections["main_content"] = "\n".join(lines[third : 2 * third])
            sections["closing"]      = "\n".join(lines[2 * third :])

    return {k: v for k, v in sections.items() if v.strip()}


def count_sections(prompt: str, model: str = "claude-sonnet") -> Dict[str, int]:
    """Return token count per detected section."""
    return {
        name: count_tokens(content, model)
        for name, content in _detect_sections(prompt).items()
    }


def section_heatmap(prompt: str, model: str = "claude-sonnet") -> Dict[str, float]:
    """Return percentage share of tokens per section (values sum to ~100)."""
    section_counts = count_sections(prompt, model)
    total = sum(section_counts.values())
    if total == 0:
        return {}
    return {
        name: round(count / total * 100, 1)
        for name, count in section_counts.items()
    }


# ---------------------------------------------------------------------------
# Output token estimation
# ---------------------------------------------------------------------------

_GENERATION_KEYWORDS = {"write", "generate", "create", "implement", "build", "code", "draft"}
_SUMMARY_KEYWORDS    = {"summarize", "summary", "tldr", "brief", "shorten"}
_EXPLAIN_KEYWORDS    = {"explain", "describe", "analyze", "review", "breakdown", "elaborate"}
_BINARY_KEYWORDS     = {"yes/no", "true/false", "is it", "does it", "can it", "will it"}
_LIST_KEYWORDS       = {"list", "enumerate", "what are", "give me", "show me"}


def estimate_output_tokens(prompt: str, model: str = "claude-sonnet") -> int:
    """Heuristically estimate how many output tokens a prompt will trigger."""
    input_tokens = count_tokens(prompt, model)
    lower = prompt.lower()

    if any(w in lower for w in _GENERATION_KEYWORDS):
        ratio = 1.5
    elif any(w in lower for w in _SUMMARY_KEYWORDS):
        ratio = 0.3
    elif any(w in lower for w in _EXPLAIN_KEYWORDS):
        ratio = 0.8
    elif any(w in lower for w in _BINARY_KEYWORDS):
        ratio = 0.1
    elif any(w in lower for w in _LIST_KEYWORDS):
        ratio = 0.6
    else:
        ratio = 0.7

    return max(50, int(input_tokens * ratio))
