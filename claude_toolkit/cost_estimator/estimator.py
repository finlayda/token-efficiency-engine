"""
Cost estimation for Claude API calls and multi-provider comparisons.

Pricing is sourced from each provider's published rates (per million tokens).
All prices in USD.

Provider coverage
-----------------
  anthropic  — Claude 4.x / 3.x series
  openai     — GPT-4o, GPT-4o-mini, GPT-4-turbo
  google     — Gemini 1.5 Pro / Flash, Gemini 2.0 Flash
  bedrock    — Amazon Bedrock pass-through for Claude / Titan
"""

from typing import Dict, List, Optional

from claude_toolkit.models import CostEstimate, ProviderComparison, ProviderCostRow

# ---------------------------------------------------------------------------
# Anthropic pricing table — update as rates change
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, Dict] = {
    "claude-opus-4-6": {
        "input_per_million":  15.00,
        "output_per_million": 75.00,
        "context_window":     200_000,
    },
    "claude-sonnet-4-6": {
        "input_per_million":  3.00,
        "output_per_million": 15.00,
        "context_window":     200_000,
    },
    "claude-sonnet-4-5": {
        "input_per_million":  3.00,
        "output_per_million": 15.00,
        "context_window":     200_000,
    },
    "claude-haiku-4-5": {
        "input_per_million":  0.80,
        "output_per_million": 4.00,
        "context_window":     200_000,
    },
    # Short aliases
    "claude-opus": {
        "input_per_million":  15.00,
        "output_per_million": 75.00,
        "context_window":     200_000,
    },
    "claude-sonnet": {
        "input_per_million":  3.00,
        "output_per_million": 15.00,
        "context_window":     200_000,
    },
    "claude-haiku": {
        "input_per_million":  0.80,
        "output_per_million": 4.00,
        "context_window":     200_000,
    },
}


def _resolve_model(model: str) -> str:
    """Map an arbitrary model string to the closest pricing key."""
    lower = model.lower()
    for key in MODEL_PRICING:
        if key in lower or lower in key:
            return key
    # Tier-based fallback
    if "opus" in lower:
        return "claude-opus"
    if "haiku" in lower:
        return "claude-haiku"
    return "claude-sonnet"


def get_pricing(model: str) -> Dict:
    """Return the pricing dict for a given model."""
    return MODEL_PRICING[_resolve_model(model)]


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet",
    custom_pricing: Optional[Dict[str, float]] = None,
) -> CostEstimate:
    """Calculate the estimated cost of a single API call."""
    pricing = custom_pricing or get_pricing(model)
    input_cost  = (input_tokens  / 1_000_000) * pricing["input_per_million"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]

    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        model=_resolve_model(model),
    )


def format_cost(cost: float) -> str:
    """Human-readable cost string that avoids unhelpful '$0.0000'."""
    if cost == 0:
        return "$0.00000"
    if cost < 0.00001:
        return f"${cost * 1_000:.4f}m"   # show in milli-dollars
    if cost < 0.001:
        return f"${cost:.6f}"
    if cost < 0.01:
        return f"${cost:.5f}"
    return f"${cost:.4f}"


def list_models() -> Dict[str, Dict]:
    """Return a copy of the full Anthropic pricing table."""
    return MODEL_PRICING.copy()


# ---------------------------------------------------------------------------
# Multi-provider pricing tables
# ---------------------------------------------------------------------------

# Each entry: {model_id: {input_per_million, output_per_million, context_window, provider}}
PROVIDER_PRICING: Dict[str, Dict[str, Dict]] = {
    "anthropic": {
        "claude-opus-4-6": {
            "input_per_million":  15.00,
            "output_per_million": 75.00,
            "context_window":     200_000,
        },
        "claude-sonnet-4-6": {
            "input_per_million":  3.00,
            "output_per_million": 15.00,
            "context_window":     200_000,
        },
        "claude-haiku-4-5": {
            "input_per_million":  0.80,
            "output_per_million": 4.00,
            "context_window":     200_000,
        },
    },
    "openai": {
        "gpt-4o": {
            "input_per_million":  2.50,
            "output_per_million": 10.00,
            "context_window":     128_000,
        },
        "gpt-4o-mini": {
            "input_per_million":  0.15,
            "output_per_million": 0.60,
            "context_window":     128_000,
        },
        "gpt-4-turbo": {
            "input_per_million":  10.00,
            "output_per_million": 30.00,
            "context_window":     128_000,
        },
        "o1": {
            "input_per_million":  15.00,
            "output_per_million": 60.00,
            "context_window":     200_000,
        },
        "o3-mini": {
            "input_per_million":  1.10,
            "output_per_million": 4.40,
            "context_window":     200_000,
        },
    },
    "google": {
        "gemini-1.5-pro": {
            "input_per_million":  1.25,   # up to 128k; 2.50 above 128k
            "output_per_million": 5.00,
            "context_window":     2_000_000,
        },
        "gemini-1.5-flash": {
            "input_per_million":  0.075,
            "output_per_million": 0.30,
            "context_window":     1_000_000,
        },
        "gemini-2.0-flash": {
            "input_per_million":  0.10,
            "output_per_million": 0.40,
            "context_window":     1_000_000,
        },
        "gemini-2.0-flash-lite": {
            "input_per_million":  0.075,
            "output_per_million": 0.30,
            "context_window":     1_000_000,
        },
    },
    "bedrock": {
        "bedrock-claude-sonnet-4-6": {
            "input_per_million":  3.00,
            "output_per_million": 15.00,
            "context_window":     200_000,
        },
        "bedrock-claude-haiku-4-5": {
            "input_per_million":  0.80,
            "output_per_million": 4.00,
            "context_window":     200_000,
        },
        "bedrock-titan-text-premier": {
            "input_per_million":  0.50,
            "output_per_million": 1.50,
            "context_window":     32_000,
        },
        "bedrock-llama-3-70b": {
            "input_per_million":  2.65,
            "output_per_million": 3.50,
            "context_window":     8_000,
        },
    },
}


def list_all_providers() -> Dict[str, Dict[str, Dict]]:
    """Return a copy of the full multi-provider pricing table."""
    return {p: dict(m) for p, m in PROVIDER_PRICING.items()}


def compare_providers(
    input_tokens: int,
    output_tokens: int,
    providers: Optional[List[str]] = None,
) -> ProviderComparison:
    """
    Compare the cost of *input_tokens* + *output_tokens* across all providers
    (or a subset specified by *providers*).

    Args:
        input_tokens:  Number of input tokens to cost.
        output_tokens: Number of output tokens to cost.
        providers:     Optional list of provider names to include
                       (e.g. ["anthropic", "openai"]).  None = all.

    Returns:
        ProviderComparison with rows sorted cheapest → most expensive.
    """
    selected = providers or list(PROVIDER_PRICING.keys())
    rows: List[ProviderCostRow] = []

    for provider in selected:
        if provider not in PROVIDER_PRICING:
            continue
        for model_id, pricing in PROVIDER_PRICING[provider].items():
            in_cost  = (input_tokens  / 1_000_000) * pricing["input_per_million"]
            out_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
            rows.append(ProviderCostRow(
                provider=provider,
                model=model_id,
                input_cost=in_cost,
                output_cost=out_cost,
                total_cost=in_cost + out_cost,
                context_window=pricing.get("context_window", 0),
            ))

    rows.sort(key=lambda r: r.total_cost)

    return ProviderComparison(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        rows=rows,
        cheapest=rows[0] if rows else ProviderCostRow("", "", 0, 0, 0, 0),
        most_expensive=rows[-1] if rows else ProviderCostRow("", "", 0, 0, 0, 0),
    )
