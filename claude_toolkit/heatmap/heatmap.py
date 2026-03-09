"""
Token heatmap rendering.

Produces text-based visualisations of where tokens are concentrated
within a prompt — both at the section level and per line.
"""

from typing import List, Tuple

from claude_toolkit.tokenizer.counter import count_tokens, section_heatmap

_BAR_WIDTH = 40
_DENSITY_THRESHOLDS = [
    (0,  ""),
    (1,  "·"),
    (5,  "▪"),
    (15, "▪▪"),
    (30, "▪▪▪"),
    (50, "▪▪▪▪ HIGH"),
]


def _density_label(tokens: int) -> str:
    label = ""
    for threshold, indicator in _DENSITY_THRESHOLDS:
        if tokens >= threshold:
            label = indicator
    return label


def render_heatmap_text(prompt: str, model: str = "claude-sonnet") -> str:
    """Render a section-level token distribution bar chart."""
    heatmap = section_heatmap(prompt, model)
    if not heatmap:
        return "  (no sections detected)"

    lines = []
    for section, pct in sorted(heatmap.items(), key=lambda x: x[1], reverse=True):
        filled = int(pct / 100 * _BAR_WIDTH)
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
        lines.append(f"  {section:<20} [{bar}] {pct:5.1f}%")

    return "\n".join(lines)


def get_token_density(
    text: str, model: str = "claude-sonnet"
) -> List[Tuple[str, int, str]]:
    """Return per-line token counts with a visual density indicator.

    Each tuple is (line_text, token_count, density_label).
    """
    result = []
    for line in text.split("\n"):
        tokens = count_tokens(line, model) if line.strip() else 0
        result.append((line, tokens, _density_label(tokens)))
    return result
