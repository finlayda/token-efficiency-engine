"""
Session log analyzer.

Parses a JSON log file of Claude prompts/responses, computes aggregate
token usage and cost, and ranks prompts by token consumption.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from claude_toolkit.models import SessionEntry, SessionSummary
from claude_toolkit.tokenizer.counter import count_tokens
from claude_toolkit.cost_estimator.estimator import estimate_cost


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _parse_entry(idx: int, item: Any) -> Optional[SessionEntry]:
    """Coerce a raw dict into a SessionEntry, returning None if unusable."""
    if not isinstance(item, dict):
        return None

    # Accept a variety of field names used by different logging conventions
    prompt = (
        item.get("prompt")
        or item.get("input")
        or item.get("user")
        or item.get("human")
        or item.get("content")
        or ""
    )
    response = (
        item.get("response")
        or item.get("output")
        or item.get("assistant")
        or item.get("completion")
        or ""
    )

    if not prompt and not response:
        return None

    model     = item.get("model", "claude-sonnet")
    timestamp = item.get("timestamp") or item.get("created_at")

    # Use provided token counts when available; otherwise estimate
    input_tokens  = item.get("input_tokens")  or item.get("prompt_tokens")
    output_tokens = item.get("output_tokens") or item.get("completion_tokens")

    if input_tokens is None:
        input_tokens = count_tokens(str(prompt), model)
    if output_tokens is None:
        output_tokens = count_tokens(str(response), model)

    return SessionEntry(
        prompt_id=idx,
        prompt=str(prompt),
        response=str(response),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        model=model,
        timestamp=timestamp,
    )


def _parse_log(data: Any) -> List[SessionEntry]:
    """Normalise any supported log shape into a flat list of SessionEntry."""
    entries: List[SessionEntry] = []

    if isinstance(data, list):
        for i, item in enumerate(data):
            e = _parse_entry(i + 1, item)
            if e:
                entries.append(e)

    elif isinstance(data, dict):
        # Try common wrapper keys first
        for key in ("messages", "prompts", "entries", "interactions", "turns", "log"):
            if key in data and isinstance(data[key], list):
                for i, item in enumerate(data[key]):
                    e = _parse_entry(i + 1, item)
                    if e:
                        entries.append(e)
                break
        else:
            # Single-entry dict
            e = _parse_entry(1, data)
            if e:
                entries.append(e)

    return entries


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_session_data(data: Any) -> SessionSummary:
    """Analyze raw session data and return an aggregate SessionSummary."""
    entries = _parse_log(data)

    if not entries:
        return SessionSummary(
            total_prompts=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            total_cost=0.0,
            top_consumers=[],
            model_breakdown={},
        )

    total_input  = sum(e.input_tokens  for e in entries)
    total_output = sum(e.output_tokens for e in entries)
    total_cost   = 0.0
    model_breakdown: Dict[str, Dict] = {}
    per_prompt: List[Dict] = []

    for entry in entries:
        cost = estimate_cost(entry.input_tokens, entry.output_tokens, entry.model)
        total_cost += cost.total_cost

        m = entry.model
        if m not in model_breakdown:
            model_breakdown[m] = {
                "prompts": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
            }
        model_breakdown[m]["prompts"]       += 1
        model_breakdown[m]["input_tokens"]  += entry.input_tokens
        model_breakdown[m]["output_tokens"] += entry.output_tokens
        model_breakdown[m]["cost"]          += cost.total_cost

        preview = entry.prompt[:80] + ("..." if len(entry.prompt) > 80 else "")
        per_prompt.append({
            "prompt_id":     entry.prompt_id,
            "prompt_preview": preview,
            "input_tokens":  entry.input_tokens,
            "output_tokens": entry.output_tokens,
            "total_tokens":  entry.input_tokens + entry.output_tokens,
            "cost":          cost.total_cost,
            "model":         entry.model,
            "timestamp":     entry.timestamp,
        })

    top_consumers = sorted(per_prompt, key=lambda x: x["total_tokens"], reverse=True)

    return SessionSummary(
        total_prompts=len(entries),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_input + total_output,
        total_cost=total_cost,
        top_consumers=top_consumers,
        model_breakdown=model_breakdown,
    )


def parse_session_entries(data: Any) -> List[SessionEntry]:
    """Parse raw session JSON/dict/list into a flat list of SessionEntry objects.

    Public wrapper around the internal _parse_log function.
    """
    return _parse_log(data)


def analyze_session_file(path: str) -> SessionSummary:
    """Load a JSON log file from *path* and return its SessionSummary."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    return analyze_session_data(data)
