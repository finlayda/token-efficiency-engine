"""
Real-time token usage monitor.

Modes
-----
1. Simulation mode (no API key required):
   Feed a pre-written response string and watch the live display update
   as if it were streaming.

2. Live API mode (requires ANTHROPIC_API_KEY + `pip install anthropic`):
   Streams a real completion from the Anthropic API and updates the live
   display token-by-token.

Usage
-----
    # Simulation
    from claude_toolkit.monitor.tracker import simulate_stream
    stats = simulate_stream("Explain gravity", "Gravity is ...", model="claude-sonnet")

    # Real API
    from claude_toolkit.monitor.tracker import monitor_from_api
    stats = monitor_from_api("Explain gravity", model="claude-sonnet-4-6")
"""

import time
from typing import Callable, Iterator, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from claude_toolkit.cost_estimator.estimator import estimate_cost, format_cost
from claude_toolkit.models import StreamStats
from claude_toolkit.tokenizer.counter import count_tokens

console = Console()


# ---------------------------------------------------------------------------
# Rich live display
# ---------------------------------------------------------------------------

def _build_live_panel(stats: StreamStats) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("Metric", style="cyan",  min_width=24)
    t.add_column("Value",  style="white", justify="right", min_width=16)

    cost = estimate_cost(stats.input_tokens, stats.output_tokens, stats.model).total_cost

    t.add_row("Model",          stats.model)
    t.add_row("Input tokens",   f"[yellow]{stats.input_tokens:,}[/yellow]")
    t.add_row("Output tokens",  f"[green]{stats.output_tokens:,}[/green]")
    t.add_row("Total tokens",   f"[bold]{stats.total_tokens:,}[/bold]")
    t.add_row("Elapsed",        f"{stats.elapsed_seconds:.1f}s")
    t.add_row("Tokens / sec",   f"{stats.tokens_per_second:.1f}")
    t.add_row("[bold]Live cost", f"[bold cyan]{format_cost(cost)}[/bold cyan]")

    return Panel(
        t,
        title="[bold cyan] Live Token Monitor [/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------

def monitor_stream(
    token_source: Iterator[str],
    input_prompt: str,
    model: str = "claude-sonnet",
    on_complete: Optional[Callable[[StreamStats], None]] = None,
) -> StreamStats:
    """
    Drive a Rich live display from an iterator of text chunks.

    Args:
        token_source:  Iterator yielding text chunks (each counted as tokens).
        input_prompt:  The original input prompt (for input token counting).
        model:         Model name used for cost estimation.
        on_complete:   Optional callback invoked with final StreamStats.

    Returns:
        Final StreamStats after the stream ends.
    """
    stats = StreamStats(
        input_tokens=count_tokens(input_prompt, model),
        output_tokens=0,
        model=model,
        elapsed_seconds=0.0,
    )
    start = time.perf_counter()

    with Live(_build_live_panel(stats), console=console, refresh_per_second=10) as live:
        for chunk in token_source:
            stats.output_tokens += count_tokens(chunk, model)
            stats.elapsed_seconds = time.perf_counter() - start
            live.update(_build_live_panel(stats))

        stats.elapsed_seconds = time.perf_counter() - start
        live.update(_build_live_panel(stats))

    if on_complete:
        on_complete(stats)

    return stats


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def simulate_stream(
    prompt: str,
    response_text: str,
    model: str = "claude-sonnet",
    tokens_per_second: float = 30.0,
) -> StreamStats:
    """
    Simulate a streaming response from a pre-written text at *tokens_per_second*.

    Useful for testing the live monitor without an API key.
    """
    words = response_text.split()
    # Emit ~10 chunks per second; each chunk = tokens_per_second / 10 words
    chunk_size = max(1, int(tokens_per_second / 10))
    delay = 1.0 / 10          # 0.1 s per chunk → 10 fps

    def _word_stream() -> Iterator[str]:
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            time.sleep(delay)
            yield chunk

    return monitor_stream(_word_stream(), prompt, model)


# ---------------------------------------------------------------------------
# Live API mode
# ---------------------------------------------------------------------------

def monitor_from_api(
    prompt: str,
    model: str = "claude-sonnet",
    max_tokens: int = 1024,
    system: Optional[str] = None,
) -> Optional[StreamStats]:
    """
    Stream a real completion from the Anthropic API and show a live monitor.

    Requirements:
      - `pip install anthropic`
      - ANTHROPIC_API_KEY environment variable set

    Returns StreamStats on success, or None if the SDK / key is unavailable.
    """
    try:
        import os
        import anthropic
    except ImportError:
        console.print(
            "[yellow]anthropic SDK not installed. "
            "Run: pip install anthropic[/yellow]"
        )
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[yellow]ANTHROPIC_API_KEY is not set. "
            "Running simulation with a placeholder response.[/yellow]"
        )
        return simulate_stream(
            prompt,
            "This is a simulated response. Set ANTHROPIC_API_KEY to use live streaming.",
            model,
        )

    # Map short aliases → valid API model IDs
    _MODEL_MAP = {
        "claude-opus":   "claude-opus-4-6",
        "claude-sonnet": "claude-sonnet-4-6",
        "claude-haiku":  "claude-haiku-4-5-20251001",
    }
    api_model = _MODEL_MAP.get(model, model)

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": api_model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system

    final_usage = None

    with client.messages.stream(**kwargs) as stream:
        def _api_chunks() -> Iterator[str]:
            for text in stream.text_stream:
                yield text

        stats = monitor_stream(_api_chunks(), prompt, model)

        try:
            msg = stream.get_final_message()
            if msg and msg.usage:
                final_usage = msg.usage
        except Exception:  # noqa: BLE001
            pass

    # Prefer API-reported token counts over local estimates
    if final_usage:
        stats.input_tokens  = final_usage.input_tokens
        stats.output_tokens = final_usage.output_tokens

    return stats
