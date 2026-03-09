"""
Prompt optimizer: applies multiple strategies to reduce token count
while preserving the original meaning and intent.
"""

import re
from typing import List, Tuple

from claude_toolkit.models import OptimizationResult, OptimizationStrategy
from claude_toolkit.tokenizer.counter import count_tokens

# ---------------------------------------------------------------------------
# Strategy 1 — Phrase compression map
# ---------------------------------------------------------------------------
# Each entry is (regex_pattern, replacement).  Patterns are applied in order.

_COMPRESSION_MAP: List[Tuple[str, str]] = [
    # Social filler
    (r"\bplease\s+",                        ""),
    (r"\bcould you please\b",               ""),
    (r"\bi would like you to\b",            ""),
    (r"\bkindly\s+",                        ""),
    (r"\bi want you to\b",                  ""),
    (r"\bcan you please\b",                 ""),
    (r"\bcould you\b",                      ""),
    (r"\bwould you\b",                      ""),
    # Verbose connectives
    (r"\bin order to\b",                    "to"),
    (r"\bfor the purpose of\b",             "to"),
    (r"\bwith the goal of\b",               "to"),
    (r"\bwith the intention of\b",          "to"),
    (r"\bin the event that\b",              "if"),
    (r"\bat this point in time\b",          "now"),
    (r"\bat the present time\b",            "now"),
    (r"\bdue to the fact that\b",           "because"),
    (r"\bin spite of the fact that\b",      "although"),
    (r"\bregardless of the fact that\b",    "although"),
    (r"\bin the case of\b",                 "for"),
    (r"\bit is important to note that\b",   "Note:"),
    (r"\bit should be noted that\b",        "Note:"),
    (r"\bplease note that\b",               "Note:"),
    (r"\bplease be aware that\b",           "Note:"),
    (r"\bas you can see\b",                 ""),
    (r"\bin other words\b",                 "i.e.,"),
    (r"\bfor example\b",                    "e.g.,"),
    (r"\bfor instance\b",                   "e.g.,"),
    (r"\bin addition to this\b",            "also,"),
    (r"\bin addition,\b",                   "also,"),
    (r"\bfurthermore,\b",                   "also,"),
    (r"\badditionally,\b",                  "also,"),
    (r"\bmoreover,\b",                      "also,"),
    (r"\bthis means that\b",                "so"),
    (r"\bit is worth mentioning that\b",    ""),
    (r"\bit is worth noting that\b",        ""),
    (r"\bwhat i want you to do is\b",       ""),
    (r"\byour task is to\b",                ""),
    (r"\byour goal is to\b",                ""),
    # Instruction verbosity
    (r"\bmake sure (?:to |that )?\b",       "ensure "),
    (r"\bplease make sure\b",               "ensure"),
    (r"\bensure that you\b",                "ensure you"),
    (r"\bremember to\b",                    ""),
    (r"\bdon't forget to\b",                ""),
    # Redundant qualifiers
    (r"\bvery unique\b",                    "unique"),
    (r"\babsolutely essential\b",           "essential"),
    (r"\bentirely complete\b",              "complete"),
    (r"\bfinal end result\b",               "result"),
    (r"\bfuture plans\b",                   "plans"),
    (r"\bpast history\b",                   "history"),
    # Verbose output instructions
    (r"\bplease provide (?:me with )?(?:a |an )?\b", "provide "),
    (r"\bgive me (?:a |an )?\b",            "give "),
    (r"\btell me (?:about )?",              ""),
    (r"\bexplain to me\b",                  "explain"),
    (r"\bwalk me through\b",                "explain"),
]

# ---------------------------------------------------------------------------
# Strategy 2 — Redundant preamble patterns (start-of-prompt)
# ---------------------------------------------------------------------------

_PREAMBLE_PATTERNS = [
    r"^(?:hi|hello|hey)[,!.]?\s+",
    r"^(?:i hope you(?:'re| are) doing well)[.!]?\s+",
    r"^(?:i need your help with|i need help with|i need you to)\s+",
    r"^(?:i'm working on|i am working on)\s+.*?(?:and|,)\s+",
    r"^(?:as an? (?:expert|senior|experienced) .*?,?\s+)",
    r"^(?:you are an? (?:expert|senior|experienced) .*?\.\s+)",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_compression(text: str) -> Tuple[str, int]:
    """Apply the phrase compression map. Returns (new_text, n_replacements)."""
    count = 0
    for pattern, replacement in _COMPRESSION_MAP:
        text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
        count += n
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), count


def _remove_redundant_preamble(text: str) -> Tuple[str, bool]:
    """Strip social filler from the beginning of a prompt."""
    for pattern in _PREAMBLE_PATTERNS:
        new_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        if new_text != text:
            return new_text.strip(), True
    return text, False


def _deduplicate_sentences(text: str) -> Tuple[str, int]:
    """Remove exact duplicate sentences (case/whitespace normalised)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen: set = set()
    unique: List[str] = []
    removed = 0
    for s in sentences:
        key = re.sub(r"\s+", " ", s.strip().lower())
        if key and key in seen:
            removed += 1
        else:
            if key:
                seen.add(key)
            unique.append(s)
    return " ".join(unique), removed


def _deduplicate_list_items(text: str) -> Tuple[str, int]:
    """Remove duplicate bullet points or numbered list lines."""
    lines = text.split("\n")
    seen: set = set()
    unique_lines: List[str] = []
    removed = 0
    for line in lines:
        key = re.sub(r"^[\s\-*•#>0-9.]+", "", line).strip().lower()
        if key and key in seen:
            removed += 1
        else:
            if key:
                seen.add(key)
            unique_lines.append(line)
    return "\n".join(unique_lines), removed


def _remove_excessive_formatting(text: str) -> Tuple[str, int]:
    """Trim redundant blank lines and trailing whitespace."""
    changes = 0
    new_text = re.sub(r"\n{3,}", "\n\n", text)
    if new_text != text:
        changes += 1
        text = new_text
    new_text = "\n".join(line.rstrip() for line in text.split("\n"))
    if new_text != text:
        changes += 1
        text = new_text
    new_text = re.sub(r"\*{3}(.*?)\*{3}", r"**\1**", text)
    if new_text != text:
        changes += 1
        text = new_text
    return text, changes


def _convert_paragraphs_to_bullets(text: str) -> Tuple[str, bool]:
    """Convert dense instruction paragraphs (3+ sentences) to bullet lists."""
    _INSTRUCTION_VERBS = re.compile(
        r"\b(?:ensure|make sure|use|add|remove|avoid|keep|set|check|verify|"
        r"include|exclude|format|write|return|output|provide)\b",
        re.IGNORECASE,
    )
    paragraphs = re.split(r"\n{2,}", text)
    changed = False
    result: List[str] = []
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!])\s+", para.strip())
        already_list = para.strip().startswith(("-", "*", "•", "1.", "2."))
        instruction_count = len(_INSTRUCTION_VERBS.findall(para))
        if len(sentences) >= 3 and not already_list and instruction_count >= 2:
            result.append("\n".join(f"- {s.strip()}" for s in sentences if s.strip()))
            changed = True
        else:
            result.append(para)
    return "\n\n".join(result), changed


def _compress_examples(text: str) -> Tuple[str, bool]:
    """Truncate very long inline examples to a 25-word preview."""
    _EXAMPLE_RE = re.compile(
        r"((?:example|e\.g\.|for example)[:\s]+)(.*?)(?=\n\n|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    def _shorten(m: re.Match) -> str:
        prefix, body = m.group(1), m.group(2)
        words = body.split()
        return prefix + (" ".join(words[:25]) + " [...]" if len(words) > 30 else body)

    new_text, n = _EXAMPLE_RE.subn(_shorten, text)
    return new_text, n > 0


# ---------------------------------------------------------------------------
# Suggestion generator
# ---------------------------------------------------------------------------

def _generate_suggestions(original: str, strategies: List[OptimizationStrategy]) -> List[str]:
    suggestions: List[str] = []
    input_tokens = count_tokens(original)

    if len(re.findall(r"(?i)\bcontext\b|\bbackground\b|\bproject\b", original)) > 3:
        suggestions.append(
            "Move repeated context to CLAUDE.md or a system prompt — avoids re-sending it on every query."
        )

    example_count = len(re.findall(r"(?i)\bexample\b|\be\.g\.\b|\bfor instance\b", original))
    if example_count > 2:
        suggestions.append(
            f"Found {example_count} examples. Reduce to 1 representative example, "
            "or move them to a referenced file."
        )

    if input_tokens > 2_000:
        suggestions.append(
            f"Prompt is {input_tokens:,} tokens. Consider splitting into smaller, focused queries."
        )

    if input_tokens > 5_000:
        suggestions.append(
            "Large prompt (5 k+ tokens). Use RAG/retrieval to inject only relevant context "
            "instead of full documents."
        )

    code_blocks = re.findall(r"```[\s\S]*?```", original)
    if sum(len(b) for b in code_blocks) > 500:
        suggestions.append(
            "Large code blocks detected. Reference files by path (@src/file.py) "
            "instead of pasting full content."
        )

    word_freq: dict = {}
    for w in original.lower().split():
        if len(w) > 5:
            word_freq[w] = word_freq.get(w, 0) + 1
    repeated = [w for w, c in word_freq.items() if c > 3]
    if repeated:
        suggestions.append(
            f"Word(s) '{', '.join(repeated[:3])}' appear frequently — "
            "define once and reference with a variable or abbreviation."
        )

    numbered = re.findall(r"^\d+\.\s+", original, re.MULTILINE)
    if len(numbered) > 6:
        suggestions.append(
            f"{len(numbered)} numbered steps found. "
            "Group related steps under headers to reduce tokens and cognitive load."
        )

    if not any(s.applied for s in strategies):
        suggestions.append(
            "This prompt is already lean. "
            "Consider prompt caching for repeated system prompts to cut costs further."
        )

    return suggestions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimize(
    prompt: str,
    model: str = "claude-sonnet",
    aggressive: bool = False,
) -> OptimizationResult:
    """Optimize *prompt* using a pipeline of token-reduction strategies.

    Args:
        prompt:     The prompt text to optimize.
        model:      Model name used for token counting.
        aggressive: When True, also restructures paragraphs and compresses
                    examples (may alter phrasing more significantly).

    Returns:
        An OptimizationResult containing the optimised prompt and metadata.
    """
    original_tokens = count_tokens(prompt, model)
    strategies: List[OptimizationStrategy] = []
    current = prompt

    def _run(name: str, description: str, fn, *args) -> None:
        nonlocal current
        before = count_tokens(current, model)
        result, changed = fn(current, *args)
        after = count_tokens(result, model)
        saved = max(0, before - after)
        applied = bool(changed)
        strategies.append(OptimizationStrategy(name=name, description=description,
                                                tokens_saved=saved, applied=applied))
        if applied:
            current = result

    _run("Remove social filler",
         "Strip greeting and preamble phrases that carry no instructional value",
         _remove_redundant_preamble)

    _run("Compress verbose phrases",
         "Replace multi-word verbose constructions with compact equivalents",
         _apply_compression)

    _run("Deduplicate sentences",
         "Remove repeated or near-identical sentences",
         _deduplicate_sentences)

    _run("Deduplicate list items",
         "Remove repeated bullet points or numbered steps",
         _deduplicate_list_items)

    _run("Remove excessive formatting",
         "Strip redundant blank lines, trailing spaces, and over-emphasis",
         _remove_excessive_formatting)

    if aggressive:
        _run("Convert paragraphs to bullets",
             "Transform dense instruction paragraphs into compact bullet lists",
             _convert_paragraphs_to_bullets)

        _run("Compress examples",
             "Truncate verbose inline examples to a 25-word representative snippet",
             _compress_examples)

    optimized_tokens = count_tokens(current, model)
    tokens_saved = max(0, original_tokens - optimized_tokens)
    reduction_pct = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0.0

    return OptimizationResult(
        original_prompt=prompt,
        optimized_prompt=current,
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
        tokens_saved=tokens_saved,
        reduction_pct=round(reduction_pct, 2),
        strategies_applied=strategies,
        suggestions=_generate_suggestions(prompt, strategies),
    )
