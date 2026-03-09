"""
CLI entry point for the Claude Token & Cost Optimization Toolkit v2.

v1 Commands
-----------
  analyze-prompt     Analyse token usage and cost for a single prompt
  optimize-prompt    Automatically reduce token count in a prompt
  analyze-session    Parse a session log and show aggregate cost/usage stats
  compare            Side-by-side comparison of two prompt files
  list-models        Show available models and their per-token pricing

v2 Commands (FinOps & Prompt Optimization Platform)
----------------------------------------------------
  detect-cache       Detect cacheable prompt prefixes across a session
  extract-context    Generate a CLAUDE.md from repeated session context
  set-budget         Configure session/daily cost budget limits
  check-budget       Evaluate a cost amount against configured budgets
  rag-analyze        Identify large context blocks suitable for RAG
  prune-examples     Cluster & prune redundant in-prompt examples
  compare-providers  Compare cost across Anthropic, OpenAI, Google, Bedrock
  monitor-session    Live token counter (simulation or real API stream)
  serve              Start the IDE integration JSON API server
"""

import difflib
import json
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from claude_toolkit.api_server.server import ToolkitApiServer
from claude_toolkit.budget.budget import (
    check_budget, get_daily_spend, get_spend_history,
    load_budget_config, save_budget_config,
)
from claude_toolkit.cache_detector.detector import detect_cache_candidates
from claude_toolkit.context_extractor.extractor import generate_claude_md
from claude_toolkit.cost_estimator.estimator import (
    compare_providers, estimate_cost, format_cost, get_pricing,
    list_all_providers, list_models,
)
from claude_toolkit.example_pruner.pruner import prune_examples
from claude_toolkit.heatmap.heatmap import get_token_density, render_heatmap_text
from claude_toolkit.models import BudgetConfig, PromptAnalysis
from claude_toolkit.monitor.tracker import monitor_from_api, simulate_stream
from claude_toolkit.prompt_optimizer.optimizer import optimize
from claude_toolkit.rag_advisor.advisor import analyze_rag_opportunities
from claude_toolkit.session_analyzer.analyzer import (
    analyze_session_file, analyze_session_data, parse_session_entries,
)
from claude_toolkit.tokenizer.counter import (
    count_tokens, estimate_output_tokens, section_heatmap,
)

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompt(source: str) -> str:
    """Read prompt from a file path or stdin when source is '-'."""
    if source == "-":
        return sys.stdin.read()
    p = Path(source)
    if not p.exists():
        raise click.ClickException(f"File not found: {source}")
    return p.read_text(encoding="utf-8")


def _build_analysis(prompt: str, model: str) -> PromptAnalysis:
    input_tokens  = count_tokens(prompt, model)
    output_tokens = estimate_output_tokens(prompt, model)
    cost          = estimate_cost(input_tokens, output_tokens, model)
    heatmap       = section_heatmap(prompt, model)
    opt           = optimize(prompt, model)          # non-destructive; used for suggestions

    return PromptAnalysis(
        prompt=prompt,
        input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        cost=cost,
        section_breakdown={
            k: max(1, int(v * input_tokens / 100)) for k, v in heatmap.items()
        },
        heatmap=heatmap,
        suggestions=opt.suggestions,
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option("2.0.0", prog_name="claude-toolkit")
def cli():
    """Claude Code LLM Token & Cost Optimization Toolkit.

    Analyse, optimise, and track token usage and cost for Claude Code
    workflows.  Run any sub-command with --help for details.
    """


# ---------------------------------------------------------------------------
# analyze-prompt
# ---------------------------------------------------------------------------

@cli.command("analyze-prompt")
@click.argument("source", default="-", metavar="FILE")
@click.option("--model", "-m", default="claude-sonnet", show_default=True,
              help="Model to price against.")
@click.option("--json", "as_json", is_flag=True,
              help="Emit results as JSON.")
@click.option("--heatmap", "show_heatmap", is_flag=True,
              help="Show per-line token-density heatmap.")
def analyze_prompt(source: str, model: str, as_json: bool, show_heatmap: bool):
    """Estimate token usage and cost for a prompt.

    SOURCE is a file path, or '-' to read from stdin.

    \b
    Examples:
      claude-toolkit analyze-prompt prompt.txt
      claude-toolkit analyze-prompt prompt.txt --model claude-opus-4-6
      echo "your prompt" | claude-toolkit analyze-prompt -
      claude-toolkit analyze-prompt prompt.txt --json
      claude-toolkit analyze-prompt prompt.txt --heatmap
    """
    prompt   = _load_prompt(source)
    analysis = _build_analysis(prompt, model)

    if as_json:
        click.echo(json.dumps({
            "model":                    model,
            "input_tokens":             analysis.input_tokens,
            "estimated_output_tokens":  analysis.estimated_output_tokens,
            "total_tokens_estimated":   analysis.input_tokens + analysis.estimated_output_tokens,
            "cost": {
                "input":  analysis.cost.input_cost,
                "output": analysis.cost.output_cost,
                "total":  analysis.cost.total_cost,
            },
            "section_heatmap":  analysis.heatmap,
            "suggestions":      analysis.suggestions,
        }, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]Prompt Analysis[/bold cyan]"))

    # ── Token & cost table ──────────────────────────────────────────────────
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Metric",                   style="cyan",  min_width=30)
    t.add_column("Value",  justify="right",  min_width=20)

    pricing = get_pricing(model)
    t.add_row("Model",                    model)
    t.add_row("Input Tokens",             f"[yellow]{analysis.input_tokens:,}[/yellow]")
    t.add_row("Est. Output Tokens",       f"{analysis.estimated_output_tokens:,}")
    t.add_row("Total Tokens (estimated)", f"{analysis.input_tokens + analysis.estimated_output_tokens:,}")
    t.add_row("─" * 30,                   "─" * 20)
    t.add_row("Input Rate",               f"${pricing['input_per_million']:.2f} / M tokens")
    t.add_row("Output Rate",              f"${pricing['output_per_million']:.2f} / M tokens")
    t.add_row("Input Cost",               format_cost(analysis.cost.input_cost))
    t.add_row("Output Cost",              format_cost(analysis.cost.output_cost))
    t.add_row("[bold]Total Estimated Cost[/bold]",
              f"[bold green]{format_cost(analysis.cost.total_cost)}[/bold green]")
    console.print(t)

    # ── Section heatmap ─────────────────────────────────────────────────────
    if analysis.heatmap:
        console.print()
        console.print("[bold]Token Distribution by Section[/bold]")
        console.print(render_heatmap_text(prompt, model))

    # ── Per-line density ────────────────────────────────────────────────────
    if show_heatmap:
        console.print()
        console.print("[bold]Per-line Token Density[/bold]")
        console.print("[dim]  {'Line':<54} Tok   Density[/dim]")
        for line_text, tokens, label in get_token_density(prompt, model):
            if not tokens:
                continue
            preview = (line_text[:52] + "..") if len(line_text) > 54 else line_text
            color   = "red" if tokens >= 50 else "yellow" if tokens >= 15 else "green"
            console.print(
                f"  [dim]{preview:<54}[/dim] [{color}]{tokens:4}[/{color}]  {label}"
            )

    # ── Suggestions ─────────────────────────────────────────────────────────
    if analysis.suggestions:
        console.print()
        console.print("[bold]Optimization Suggestions[/bold]")
        for i, s in enumerate(analysis.suggestions, 1):
            console.print(f"  [cyan]{i}.[/cyan] {s}")

    console.print()


# ---------------------------------------------------------------------------
# optimize-prompt
# ---------------------------------------------------------------------------

@cli.command("optimize-prompt")
@click.argument("source", default="-", metavar="FILE")
@click.option("--model", "-m",       default="claude-sonnet", show_default=True,
              help="Model to price/count against.")
@click.option("--aggressive", "-a",  is_flag=True,
              help="Also restructure paragraphs and compress examples.")
@click.option("--json", "as_json",   is_flag=True,
              help="Emit results as JSON.")
@click.option("--output", "-o",      default=None, metavar="FILE",
              help="Write the optimised prompt to FILE.")
@click.option("--show-diff",         is_flag=True,
              help="Print a unified diff of the changes.")
def optimize_prompt(
    source: str, model: str, aggressive: bool,
    as_json: bool, output: Optional[str], show_diff: bool,
):
    """Reduce token count in a prompt automatically.

    SOURCE is a file path, or '-' to read from stdin.

    \b
    Examples:
      claude-toolkit optimize-prompt prompt.txt
      claude-toolkit optimize-prompt prompt.txt --aggressive
      claude-toolkit optimize-prompt prompt.txt -o optimized.txt
      claude-toolkit optimize-prompt prompt.txt --show-diff
      claude-toolkit optimize-prompt prompt.txt --json
    """
    prompt = _load_prompt(source)
    result = optimize(prompt, model, aggressive=aggressive)

    orig_out = estimate_output_tokens(prompt,                  model)
    opt_out  = estimate_output_tokens(result.optimized_prompt, model)
    orig_cost = estimate_cost(result.original_tokens,  orig_out, model)
    opt_cost  = estimate_cost(result.optimized_tokens, opt_out,  model)
    cost_saved = orig_cost.total_cost - opt_cost.total_cost

    if as_json:
        click.echo(json.dumps({
            "original_tokens":  result.original_tokens,
            "optimized_tokens": result.optimized_tokens,
            "tokens_saved":     result.tokens_saved,
            "reduction_pct":    result.reduction_pct,
            "original_cost":    orig_cost.total_cost,
            "optimized_cost":   opt_cost.total_cost,
            "cost_saved":       cost_saved,
            "strategies": [
                {"name": s.name, "applied": s.applied, "tokens_saved": s.tokens_saved}
                for s in result.strategies_applied
            ],
            "suggestions":       result.suggestions,
            "optimized_prompt":  result.optimized_prompt,
        }, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]Prompt Optimization[/bold cyan]"))

    # ── Cost comparison table ───────────────────────────────────────────────
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Metric",                    style="cyan",  min_width=26)
    t.add_column("Original",  justify="right", min_width=16)
    t.add_column("Optimized", justify="right", min_width=16)
    t.add_column("Saved",     justify="right", min_width=16)

    pct = result.reduction_pct
    t.add_row(
        "Input Tokens",
        f"[red]{result.original_tokens:,}[/red]",
        f"[green]{result.optimized_tokens:,}[/green]",
        f"[bold]{result.tokens_saved:,} ({pct:.1f}%)[/bold]",
    )
    t.add_row(
        "Est. Output Tokens",
        f"{orig_out:,}", f"{opt_out:,}",
        f"{orig_out - opt_out:,}",
    )
    t.add_row(
        "Input Cost",
        format_cost(orig_cost.input_cost), format_cost(opt_cost.input_cost),
        format_cost(orig_cost.input_cost - opt_cost.input_cost),
    )
    t.add_row(
        "Output Cost",
        format_cost(orig_cost.output_cost), format_cost(opt_cost.output_cost),
        format_cost(orig_cost.output_cost - opt_cost.output_cost),
    )
    t.add_row(
        "[bold]Total Cost[/bold]",
        f"[bold red]{format_cost(orig_cost.total_cost)}[/bold red]",
        f"[bold green]{format_cost(opt_cost.total_cost)}[/bold green]",
        f"[bold cyan]{format_cost(cost_saved)}[/bold cyan]",
    )
    console.print(t)

    # ── Strategies ──────────────────────────────────────────────────────────
    console.print()
    console.print("[bold]Optimization Strategies Applied[/bold]")
    st = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    st.add_column("Strategy",     style="cyan")
    st.add_column("Applied",      justify="center", min_width=8)
    st.add_column("Tokens Saved", justify="right",  min_width=14)
    st.add_column("Description",  style="dim")

    for s in result.strategies_applied:
        tick  = "[green]✓[/green]" if s.applied else "[dim]–[/dim]"
        saved = f"[green]{s.tokens_saved}[/green]" if s.tokens_saved > 0 else "0"
        st.add_row(s.name, tick, saved, s.description)
    console.print(st)

    # ── Diff ────────────────────────────────────────────────────────────────
    if show_diff:
        console.print()
        console.print("[bold]Diff (original → optimized)[/bold]")
        orig_lines = result.original_prompt.splitlines(keepends=True)
        opt_lines  = result.optimized_prompt.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            orig_lines, opt_lines, fromfile="original", tofile="optimized", lineterm=""
        ))
        for line in diff:
            if   line.startswith("+") and not line.startswith("+++"):
                console.print(f"[green]{line.rstrip()}[/green]")
            elif line.startswith("-") and not line.startswith("---"):
                console.print(f"[red]{line.rstrip()}[/red]")
            elif line.startswith("@@"):
                console.print(f"[cyan]{line.rstrip()}[/cyan]")
            else:
                console.print(f"[dim]{line.rstrip()}[/dim]")

    # ── Optimized prompt preview ────────────────────────────────────────────
    console.print()
    console.print(Panel(
        result.optimized_prompt,
        title="[bold green]Optimized Prompt[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # ── Suggestions ─────────────────────────────────────────────────────────
    if result.suggestions:
        console.print()
        console.print("[bold]Further Suggestions[/bold]")
        for i, s in enumerate(result.suggestions, 1):
            console.print(f"  [cyan]{i}.[/cyan] {s}")

    # ── Write output file ───────────────────────────────────────────────────
    if output:
        Path(output).write_text(result.optimized_prompt, encoding="utf-8")
        console.print()
        console.print(f"[green]✓[/green] Optimised prompt written to [bold]{output}[/bold]")

    console.print()


# ---------------------------------------------------------------------------
# analyze-session
# ---------------------------------------------------------------------------

@cli.command("analyze-session")
@click.argument("log_file", metavar="LOG_FILE")
@click.option("--top", "-n", default=5, show_default=True,
              help="Number of top token consumers to display.")
@click.option("--json", "as_json", is_flag=True,
              help="Emit results as JSON.")
def analyze_session(log_file: str, top: int, as_json: bool):
    """Analyse a Claude Code session log file.

    LOG_FILE must be a JSON file containing prompts and/or responses.

    \b
    Examples:
      claude-toolkit analyze-session session.json
      claude-toolkit analyze-session session.json --top 10
      claude-toolkit analyze-session session.json --json
    """
    try:
        summary = analyze_session_file(log_file)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON in log file: {exc}")

    if as_json:
        click.echo(json.dumps({
            "total_prompts":       summary.total_prompts,
            "total_input_tokens":  summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "total_tokens":        summary.total_tokens,
            "total_cost":          summary.total_cost,
            "top_consumers":       summary.top_consumers[:top],
            "model_breakdown":     summary.model_breakdown,
        }, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]Session Analysis[/bold cyan]"))

    # ── Summary panel ───────────────────────────────────────────────────────
    summary_text = (
        f"[bold]Total Prompts:[/bold]   {summary.total_prompts:,}\n"
        f"[bold]Input Tokens:[/bold]    [yellow]{summary.total_input_tokens:,}[/yellow]\n"
        f"[bold]Output Tokens:[/bold]   {summary.total_output_tokens:,}\n"
        f"[bold]Total Tokens:[/bold]    [yellow]{summary.total_tokens:,}[/yellow]\n"
        f"[bold]Estimated Cost:[/bold]  [bold green]{format_cost(summary.total_cost)}[/bold green]"
    )
    console.print(Panel(summary_text, title="[bold]Session Summary[/bold]",
                        border_style="cyan", padding=(1, 2)))

    # ── Per-model breakdown ─────────────────────────────────────────────────
    if len(summary.model_breakdown) > 1:
        console.print()
        console.print("[bold]Model Breakdown[/bold]")
        mt = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        mt.add_column("Model",         style="cyan")
        mt.add_column("Prompts",       justify="right")
        mt.add_column("Input Tokens",  justify="right")
        mt.add_column("Output Tokens", justify="right")
        mt.add_column("Cost",          justify="right")
        for model_name, stats in summary.model_breakdown.items():
            mt.add_row(
                model_name,
                str(stats["prompts"]),
                f"{stats['input_tokens']:,}",
                f"{stats['output_tokens']:,}",
                f"[green]{format_cost(stats['cost'])}[/green]",
            )
        console.print(mt)

    # ── Top consumers ───────────────────────────────────────────────────────
    n_show = min(top, len(summary.top_consumers))
    console.print()
    console.print(f"[bold]Top {n_show} Token Consumers[/bold]")
    ct = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    ct.add_column("#",              justify="right",  style="dim",  min_width=3)
    ct.add_column("Prompt Preview",                              min_width=44)
    ct.add_column("Input",          justify="right",              min_width=8)
    ct.add_column("Output",         justify="right",              min_width=8)
    ct.add_column("Total",          justify="right",              min_width=8)
    ct.add_column("Cost",           justify="right",              min_width=10)

    for i, entry in enumerate(summary.top_consumers[:top], 1):
        color = "red" if i == 1 else "yellow" if i == 2 else "white"
        ct.add_row(
            f"[{color}]{i}[/{color}]",
            f"[dim]#{entry['prompt_id']}[/dim] {entry['prompt_preview']}",
            f"{entry['input_tokens']:,}",
            f"{entry['output_tokens']:,}",
            f"[{color}]{entry['total_tokens']:,}[/{color}]",
            f"[green]{format_cost(entry['cost'])}[/green]",
        )
    console.print(ct)
    console.print()


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

@cli.command("compare")
@click.argument("original_file", metavar="ORIGINAL")
@click.argument("optimized_file", metavar="OPTIMIZED")
@click.option("--model", "-m", default="claude-sonnet", show_default=True)
@click.option("--json", "as_json", is_flag=True)
def compare(original_file: str, optimized_file: str, model: str, as_json: bool):
    """Side-by-side cost comparison of two prompt files.

    \b
    Examples:
      claude-toolkit compare original.txt optimized.txt
      claude-toolkit compare v1.txt v2.txt --model claude-opus-4-6
    """
    original  = _load_prompt(original_file)
    optimized = _load_prompt(optimized_file)

    orig_in  = count_tokens(original,  model)
    opt_in   = count_tokens(optimized, model)
    orig_out = estimate_output_tokens(original,  model)
    opt_out  = estimate_output_tokens(optimized, model)
    orig_cost = estimate_cost(orig_in, orig_out, model)
    opt_cost  = estimate_cost(opt_in,  opt_out,  model)

    tokens_saved = orig_in - opt_in
    cost_saved   = orig_cost.total_cost - opt_cost.total_cost
    pct          = (tokens_saved / orig_in * 100) if orig_in > 0 else 0.0

    if as_json:
        click.echo(json.dumps({
            "original":  {"input_tokens": orig_in, "output_tokens": orig_out,
                          "cost": orig_cost.total_cost},
            "optimized": {"input_tokens": opt_in, "output_tokens": opt_out,
                          "cost": opt_cost.total_cost},
            "savings":   {"tokens": tokens_saved, "cost": cost_saved,
                          "reduction_pct": round(pct, 2)},
        }, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]Prompt Comparison[/bold cyan]"))

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Metric",           style="cyan",  min_width=24)
    t.add_column("Original",  justify="right", min_width=16)
    t.add_column("Optimized", justify="right", min_width=16)
    t.add_column("Δ Saved",   justify="right", min_width=16)

    t.add_row(
        "Input Tokens",
        f"[red]{orig_in:,}[/red]",
        f"[green]{opt_in:,}[/green]",
        f"[bold]{tokens_saved:,}  ({pct:.1f}%)[/bold]",
    )
    t.add_row(
        "Est. Output Tokens",
        f"{orig_out:,}", f"{opt_out:,}", f"{orig_out - opt_out:,}",
    )
    t.add_row(
        "[bold]Total Cost[/bold]",
        f"[bold red]{format_cost(orig_cost.total_cost)}[/bold red]",
        f"[bold green]{format_cost(opt_cost.total_cost)}[/bold green]",
        f"[bold cyan]{format_cost(cost_saved)}[/bold cyan]",
    )
    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# list-models
# ---------------------------------------------------------------------------

@cli.command("list-models")
@click.option("--json", "as_json", is_flag=True)
def list_models_cmd(as_json: bool):
    """List available models and their per-token pricing."""
    models = list_models()

    if as_json:
        click.echo(json.dumps(models, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]Available Models & Pricing[/bold cyan]"))

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Model",                   style="cyan")
    t.add_column("Input  ($/M tokens)",  justify="right")
    t.add_column("Output ($/M tokens)", justify="right")
    t.add_column("Context Window",       justify="right")

    seen: set = set()
    for model_name, pricing in models.items():
        key = (pricing.get("input_per_million"), pricing.get("output_per_million"))
        if key in seen:
            continue
        seen.add(key)
        ctx = pricing.get("context_window", "N/A")
        ctx_str = f"{ctx:,}" if isinstance(ctx, int) else str(ctx)
        t.add_row(
            model_name,
            f"${pricing['input_per_million']:.2f}",
            f"${pricing['output_per_million']:.2f}",
            ctx_str,
        )
    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# detect-cache
# ---------------------------------------------------------------------------

@cli.command("detect-cache")
@click.argument("log_file", metavar="LOG_FILE")
@click.option("--min-tokens", default=50, show_default=True,
              help="Minimum shared prefix size (tokens) to flag.")
@click.option("--threshold", default=0.6, show_default=True,
              help="Similarity threshold for grouping (0.0–1.0).")
@click.option("--json", "as_json", is_flag=True)
def detect_cache(log_file: str, min_tokens: int, threshold: float, as_json: bool):
    """Detect cacheable prompt prefixes across a session log.

    Reads a JSON session log and identifies shared prefixes or similar
    opening blocks that could be moved to a stable system prompt and
    cached via Anthropic's prompt-caching API.

    \b
    Examples:
      claude-toolkit detect-cache session.json
      claude-toolkit detect-cache session.json --min-tokens 100
      claude-toolkit detect-cache session.json --json
    """
    import json as _json
    from pathlib import Path as _Path

    try:
        raw = _json.loads(_Path(log_file).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise click.ClickException(f"File not found: {log_file}")
    except _json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON: {exc}")

    entries = parse_session_entries(raw)
    pairs = [(e.prompt_id, e.prompt) for e in entries if e.prompt.strip()]

    report = detect_cache_candidates(pairs, min_tokens, threshold)

    if as_json:
        click.echo(_json.dumps({
            "total_prompts":          report.total_prompts,
            "total_cacheable_tokens": report.total_cacheable_tokens,
            "overall_savings_pct":    report.overall_savings_pct,
            "suggestion":             report.suggestion,
            "candidates": [
                {
                    "prefix_preview":         c.prefix[:120],
                    "token_count":            c.token_count,
                    "frequency":              c.frequency,
                    "estimated_tokens_saved": c.estimated_tokens_saved,
                    "savings_pct":            c.savings_pct,
                }
                for c in report.candidates
            ],
        }, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]Prompt Cache Detection[/bold cyan]"))

    color = "green" if report.overall_savings_pct < 30 else "yellow" if report.overall_savings_pct < 60 else "red"
    summary_text = (
        f"[bold]Total Prompts:[/bold]      {report.total_prompts:,}\n"
        f"[bold]Cacheable Tokens:[/bold]   [yellow]{report.total_cacheable_tokens:,}[/yellow]\n"
        f"[bold]Est. Savings:[/bold]       [{color}]{report.overall_savings_pct:.1f}%[/{color}]"
    )
    console.print(Panel(summary_text, title="[bold]Cache Opportunity Summary[/bold]",
                        border_style="cyan", padding=(1, 2)))

    if report.candidates:
        console.print()
        console.print("[bold]Cache Candidates[/bold]")
        t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        t.add_column("#",             justify="right",  style="dim",   min_width=3)
        t.add_column("Shared Prefix (preview)",                        min_width=50)
        t.add_column("Tokens",        justify="right",                 min_width=8)
        t.add_column("Frequency",     justify="right",                 min_width=10)
        t.add_column("Est. Saved",    justify="right",                 min_width=12)
        t.add_column("Savings %",     justify="right",                 min_width=10)

        for i, c in enumerate(report.candidates, 1):
            preview = c.prefix[:50] + ("..." if len(c.prefix) > 50 else "")
            t.add_row(
                str(i), f"[dim]{preview}[/dim]",
                f"[yellow]{c.token_count:,}[/yellow]",
                str(c.frequency),
                f"[green]{c.estimated_tokens_saved:,}[/green]",
                f"[bold]{c.savings_pct:.1f}%[/bold]",
            )
        console.print(t)

    console.print()
    console.print("[bold]Suggestion[/bold]")
    console.print(f"  [cyan]{report.suggestion}[/cyan]")
    console.print()


# ---------------------------------------------------------------------------
# extract-context
# ---------------------------------------------------------------------------

@cli.command("extract-context")
@click.argument("log_file", metavar="LOG_FILE")
@click.option("--output", "-o", default="CLAUDE.md", show_default=True,
              help="Path to write the generated CLAUDE.md file.")
@click.option("--project", default="Project", show_default=True,
              help="Project name for the CLAUDE.md header.")
@click.option("--min-occurrences", default=2, show_default=True,
              help="Minimum times a phrase must appear to be included.")
@click.option("--dry-run", is_flag=True,
              help="Print generated content without writing to disk.")
@click.option("--json", "as_json", is_flag=True)
def extract_context(
    log_file: str, output: str, project: str,
    min_occurrences: int, dry_run: bool, as_json: bool,
):
    """Generate a CLAUDE.md from repeated context in a session log.

    Detects phrases and sentences repeated across prompts and drafts a
    CLAUDE.md shared-context file that eliminates per-prompt repetition.

    \b
    Examples:
      claude-toolkit extract-context session.json
      claude-toolkit extract-context session.json -o CLAUDE.md --project MyApp
      claude-toolkit extract-context session.json --dry-run
    """
    import json as _json
    from pathlib import Path as _Path
    from claude_toolkit.session_analyzer.analyzer import _parse_log

    try:
        raw = _json.loads(_Path(log_file).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise click.ClickException(f"File not found: {log_file}")
    except _json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON: {exc}")

    entries = _parse_log(raw)
    prompts = [e.prompt for e in entries if e.prompt.strip()]

    out_path = None if dry_run else output
    result = generate_claude_md(prompts, project_name=project,
                                output_path=out_path, min_occurrences=min_occurrences)

    if as_json:
        click.echo(_json.dumps({
            "blocks_found":                len(result.blocks),
            "total_repeated_tokens":       result.total_repeated_tokens,
            "estimated_savings_per_prompt": result.estimated_savings_per_prompt,
            "output_path":                 output if not dry_run else None,
            "blocks": [
                {"category": b.category, "token_count": b.token_count,
                 "occurrences": b.occurrences, "content_preview": b.content[:80]}
                for b in result.blocks
            ],
        }, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]CLAUDE.md Context Extractor[/bold cyan]"))

    if not result.blocks:
        console.print(
            Panel("[yellow]No repeated context detected in this session.[/yellow]\n"
                  "The generated CLAUDE.md contains a blank template.",
                  border_style="yellow", padding=(1, 2))
        )
    else:
        t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        t.add_column("Category",    style="cyan",     min_width=16)
        t.add_column("Content (preview)",             min_width=48)
        t.add_column("Tokens",      justify="right",  min_width=8)
        t.add_column("Occurrences", justify="right",  min_width=12)

        for b in result.blocks:
            preview = b.content[:60] + ("..." if len(b.content) > 60 else "")
            t.add_row(
                b.category,
                f"[dim]{preview}[/dim]",
                f"[yellow]{b.token_count:,}[/yellow]",
                str(b.occurrences),
            )
        console.print(t)

        console.print()
        console.print(
            f"  Est. savings per prompt: "
            f"[bold green]{result.estimated_savings_per_prompt:,} tokens[/bold green]"
        )

    if dry_run:
        console.print()
        console.print(Panel(
            result.generated_content,
            title="[bold green]Generated CLAUDE.md (dry run)[/bold green]",
            border_style="green", padding=(1, 2),
        ))
    else:
        console.print()
        console.print(
            f"[green]✓[/green] CLAUDE.md written to [bold]{output}[/bold]"
        )
    console.print()


# ---------------------------------------------------------------------------
# set-budget
# ---------------------------------------------------------------------------

@cli.command("set-budget")
@click.option("--session", "session_budget", type=float, default=None,
              help="Session cost limit in USD (e.g. 5.00).")
@click.option("--daily", "daily_budget", type=float, default=None,
              help="Daily cost limit in USD (e.g. 20.00).")
@click.option("--warn-at", "warn_at_pct", type=float, default=None,
              help="Warn when spend reaches this %% of budget (default 80).")
@click.option("--clear", is_flag=True, help="Clear all saved budget settings.")
def set_budget(
    session_budget: Optional[float],
    daily_budget: Optional[float],
    warn_at_pct: Optional[float],
    clear: bool,
):
    """Configure session and daily cost budget limits.

    Settings are saved to ~/.claude-toolkit/budget.json and respected by
    all subsequent commands that produce cost estimates.

    \b
    Examples:
      claude-toolkit set-budget --session 5.00 --daily 20.00
      claude-toolkit set-budget --warn-at 70
      claude-toolkit set-budget --clear
    """
    from claude_toolkit.budget.budget import _BUDGET_FILE

    if clear:
        if _BUDGET_FILE.exists():
            _BUDGET_FILE.unlink()
        console.print("[green]✓[/green] Budget configuration cleared.")
        return

    cfg = load_budget_config(session_budget, daily_budget, warn_at_pct)
    save_budget_config(cfg)

    console.print()
    console.print(Rule("[bold cyan]Budget Configuration[/bold cyan]"))
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Setting",   style="cyan",  min_width=24)
    t.add_column("Value",     justify="right", min_width=16)

    t.add_row("Session Budget",
              f"[green]${cfg.session_budget:.2f}[/green]" if cfg.session_budget else "[dim]not set[/dim]")
    t.add_row("Daily Budget",
              f"[green]${cfg.daily_budget:.2f}[/green]" if cfg.daily_budget else "[dim]not set[/dim]")
    t.add_row("Warn At",  f"{cfg.warn_at_pct:.0f}%")
    t.add_row("Config File", str(_BUDGET_FILE))
    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# check-budget
# ---------------------------------------------------------------------------

@cli.command("check-budget")
@click.argument("cost", type=float, metavar="COST")
@click.option("--session", "session_budget", type=float, default=None)
@click.option("--daily",   "daily_budget",   type=float, default=None)
@click.option("--record",  is_flag=True,
              help="Record COST in today's daily spend tracker.")
@click.option("--history", is_flag=True,
              help="Show 7-day spend history.")
@click.option("--json", "as_json", is_flag=True)
def check_budget_cmd(
    cost: float,
    session_budget: Optional[float],
    daily_budget: Optional[float],
    record: bool,
    history: bool,
    as_json: bool,
):
    """Evaluate a cost amount against configured budget limits.

    COST is the session spend in USD (e.g. 1.23).

    \b
    Examples:
      claude-toolkit check-budget 1.23
      claude-toolkit check-budget 4.50 --session 5.00 --daily 20.00
      claude-toolkit check-budget 1.00 --record
      claude-toolkit check-budget 0 --history
    """
    import json as _json

    cfg = load_budget_config(session_budget, daily_budget)
    alert = check_budget(cost, cfg, add_to_daily=record)

    if as_json:
        click.echo(_json.dumps({
            "level":          alert.level,
            "message":        alert.message,
            "session_cost":   alert.session_cost,
            "session_budget": alert.session_budget,
            "session_pct":    alert.session_pct,
            "daily_cost":     alert.daily_cost,
            "daily_budget":   alert.daily_budget,
            "daily_pct":      alert.daily_pct,
        }, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]Budget Check[/bold cyan]"))

    level_colors = {"ok": "green", "warn": "yellow", "critical": "red", "exceeded": "bold red"}
    level_icons  = {"ok": "[OK]", "warn": "[WARN]", "critical": "[CRIT]", "exceeded": "[OVER]"}
    color = level_colors.get(alert.level, "white")
    icon  = level_icons.get(alert.level, "[?]")

    console.print(f"\n  [{color}]{icon}  {alert.message}[/{color}]\n")

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Budget",   style="cyan",  min_width=20)
    t.add_column("Spend",    justify="right", min_width=12)
    t.add_column("Limit",    justify="right", min_width=12)
    t.add_column("Used",     justify="right", min_width=10)

    if alert.session_budget is not None:
        spct = f"{alert.session_pct:.0f}%" if alert.session_pct is not None else "—"
        t.add_row("Session",
                  format_cost(alert.session_cost),
                  f"${alert.session_budget:.2f}",
                  f"[{color}]{spct}[/{color}]")
    if alert.daily_budget is not None:
        dpct = f"{alert.daily_pct:.0f}%" if alert.daily_pct is not None else "—"
        t.add_row("Daily",
                  format_cost(alert.daily_cost),
                  f"${alert.daily_budget:.2f}",
                  f"[{color}]{dpct}[/{color}]")
    console.print(t)

    if history:
        console.print()
        console.print("[bold]7-Day Spend History[/bold]")
        ht = Table(box=box.SIMPLE, show_header=True, header_style="dim")
        ht.add_column("Date",  style="cyan",  min_width=12)
        ht.add_column("Spend", justify="right", min_width=12)
        for d, s in get_spend_history(7).items():
            ht.add_row(d, f"[green]{format_cost(s)}[/green]" if s > 0 else "[dim]$0.00[/dim]")
        console.print(ht)

    console.print()


# ---------------------------------------------------------------------------
# rag-analyze
# ---------------------------------------------------------------------------

@cli.command("rag-analyze")
@click.argument("source", default="-", metavar="FILE")
@click.option("--threshold", default=300, show_default=True,
              help="Token threshold to flag a block as large context.")
@click.option("--model", "-m", default="claude-sonnet", show_default=True)
@click.option("--json", "as_json", is_flag=True)
def rag_analyze(source: str, threshold: int, model: str, as_json: bool):
    """Detect large static context blocks and recommend RAG compression.

    Identifies contiguous blocks that exceed the token threshold and
    suggests vector-retrieval strategies to shrink the prompt.

    \b
    Examples:
      claude-toolkit rag-analyze prompt.txt
      claude-toolkit rag-analyze prompt.txt --threshold 200
      cat big_prompt.txt | claude-toolkit rag-analyze -
    """
    import json as _json

    prompt = _load_prompt(source)
    advice = analyze_rag_opportunities(prompt, threshold)

    if as_json:
        click.echo(_json.dumps({
            "total_prompt_tokens":         advice.total_prompt_tokens,
            "total_context_tokens":        advice.total_context_tokens,
            "context_fraction_pct":        advice.context_fraction,
            "total_estimated_savings_pct": advice.total_estimated_savings_pct,
            "blocks": [
                {
                    "type":                  b.block_type,
                    "lines":                 f"{b.start_line}–{b.end_line}",
                    "token_count":           b.token_count,
                    "estimated_savings_pct": b.estimated_savings_pct,
                    "suggestion":            b.retrieval_suggestion,
                }
                for b in advice.blocks
            ],
            "recommendations": advice.recommendations,
        }, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]RAG Context Analysis[/bold cyan]"))

    pct = advice.total_estimated_savings_pct
    color = "green" if pct < 20 else "yellow" if pct < 50 else "red"
    summary_text = (
        f"[bold]Total Prompt Tokens:[/bold]    [yellow]{advice.total_prompt_tokens:,}[/yellow]\n"
        f"[bold]Large Context Tokens:[/bold]   [yellow]{advice.total_context_tokens:,}[/yellow]"
        f"  ({advice.context_fraction:.0f}% of prompt)\n"
        f"[bold]Est. RAG Savings:[/bold]       [{color}]{pct:.0f}%[/{color}]"
    )
    console.print(Panel(summary_text, title="[bold]Context Overview[/bold]",
                        border_style="cyan", padding=(1, 2)))

    if advice.blocks:
        console.print()
        console.print("[bold]Large Context Blocks[/bold]")
        t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        t.add_column("#",        justify="right", style="dim", min_width=3)
        t.add_column("Type",     style="cyan",                 min_width=10)
        t.add_column("Lines",    justify="right",              min_width=12)
        t.add_column("Tokens",   justify="right",              min_width=8)
        t.add_column("Savings",  justify="right",              min_width=8)
        t.add_column("Suggestion (preview)",                   min_width=40)

        for i, b in enumerate(advice.blocks, 1):
            t.add_row(
                str(i), b.block_type,
                f"{b.start_line}–{b.end_line}",
                f"[red]{b.token_count:,}[/red]",
                f"[green]~{b.estimated_savings_pct:.0f}%[/green]",
                f"[dim]{b.retrieval_suggestion[:60]}...[/dim]",
            )
        console.print(t)

    console.print()
    console.print("[bold]Recommendations[/bold]")
    for i, r in enumerate(advice.recommendations, 1):
        console.print(f"  [cyan]{i}.[/cyan] {r}")
    console.print()


# ---------------------------------------------------------------------------
# prune-examples
# ---------------------------------------------------------------------------

@cli.command("prune-examples")
@click.argument("source", default="-", metavar="FILE")
@click.option("--threshold", default=0.65, show_default=True,
              help="Similarity threshold to treat two examples as redundant (0–1).")
@click.option("--model", "-m", default="claude-sonnet", show_default=True)
@click.option("--json", "as_json", is_flag=True)
def prune_examples_cmd(source: str, threshold: float, model: str, as_json: bool):
    """Cluster and prune redundant in-prompt examples.

    Uses TF-IDF cosine similarity to group similar examples and keeps
    only the most representative member of each cluster.

    \b
    Examples:
      claude-toolkit prune-examples prompt.txt
      claude-toolkit prune-examples prompt.txt --threshold 0.70
    """
    import json as _json

    prompt = _load_prompt(source)
    result = prune_examples(prompt, threshold, model)

    if as_json:
        click.echo(_json.dumps({
            "original_examples":   len(result.original_examples),
            "kept_examples":       len(result.kept_examples),
            "clusters":            len(result.clusters),
            "original_tokens":     result.original_tokens,
            "pruned_tokens":       result.pruned_tokens,
            "tokens_saved":        result.tokens_saved,
            "reduction_pct":       result.reduction_pct,
            "cluster_detail": [
                {
                    "representative":  c.representative[:80],
                    "members":         len(c.members),
                    "similarity":      c.similarity_score,
                    "tokens_saved":    c.tokens_saved,
                }
                for c in result.clusters
            ],
        }, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]Example Pruning[/bold cyan]"))

    if not result.original_examples:
        console.print(Panel(
            "[yellow]No examples detected in this prompt.[/yellow]",
            border_style="yellow", padding=(1, 2),
        ))
        console.print()
        return

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Metric",    style="cyan",  min_width=28)
    t.add_column("Value",     justify="right", min_width=16)

    t.add_row("Examples found",   str(len(result.original_examples)))
    t.add_row("Clusters formed",  str(len(result.clusters)))
    t.add_row("Examples kept",    f"[green]{len(result.kept_examples)}[/green]")
    t.add_row("Examples pruned",
              f"[red]{len(result.original_examples) - len(result.kept_examples)}[/red]")
    t.add_row("Tokens saved",
              f"[bold green]{result.tokens_saved:,} ({result.reduction_pct:.1f}%)[/bold green]")
    console.print(t)

    if result.clusters:
        console.print()
        console.print("[bold]Cluster Detail[/bold]")
        ct = Table(box=box.SIMPLE, show_header=True, header_style="dim")
        ct.add_column("#",           justify="right", style="dim", min_width=3)
        ct.add_column("Representative (preview)",                  min_width=50)
        ct.add_column("Members",     justify="right",              min_width=8)
        ct.add_column("Similarity",  justify="right",              min_width=10)
        ct.add_column("Tokens Saved", justify="right",             min_width=12)

        for i, c in enumerate(result.clusters, 1):
            rep = c.representative[:50] + ("..." if len(c.representative) > 50 else "")
            ct.add_row(
                str(i), f"[dim]{rep}[/dim]",
                str(len(c.members)),
                f"{c.similarity_score:.2f}",
                f"[green]{c.tokens_saved:,}[/green]",
            )
        console.print(ct)
    console.print()


# ---------------------------------------------------------------------------
# compare-providers
# ---------------------------------------------------------------------------

@cli.command("compare-providers")
@click.argument("source", default="-", metavar="FILE")
@click.option("--model", "-m", default="claude-sonnet", show_default=True,
              help="Model for input token counting.")
@click.option("--providers", "-p", default=None,
              help="Comma-separated provider list (anthropic,openai,google,bedrock).")
@click.option("--top", default=10, show_default=True,
              help="Number of rows to show (sorted cheapest first).")
@click.option("--json", "as_json", is_flag=True)
def compare_providers_cmd(
    source: str, model: str, providers: Optional[str], top: int, as_json: bool,
):
    """Compare prompt cost across Anthropic, OpenAI, Google, and Bedrock.

    Counts tokens in SOURCE, estimates output tokens, then prices the call
    against every model in the selected providers.

    \b
    Examples:
      claude-toolkit compare-providers prompt.txt
      claude-toolkit compare-providers prompt.txt --providers anthropic,openai
      claude-toolkit compare-providers prompt.txt --top 5 --json
    """
    import json as _json

    prompt = _load_prompt(source)
    input_tok  = count_tokens(prompt, model)
    output_tok = estimate_output_tokens(prompt, model)

    provider_list: Optional[List[str]] = (
        [p.strip() for p in providers.split(",")] if providers else None
    )
    comparison = compare_providers(input_tok, output_tok, provider_list)

    if as_json:
        click.echo(_json.dumps({
            "input_tokens":  comparison.input_tokens,
            "output_tokens": comparison.output_tokens,
            "cheapest": {
                "provider": comparison.cheapest.provider,
                "model":    comparison.cheapest.model,
                "cost":     comparison.cheapest.total_cost,
            },
            "rows": [
                {
                    "provider":       r.provider,
                    "model":          r.model,
                    "input_cost":     r.input_cost,
                    "output_cost":    r.output_cost,
                    "total_cost":     r.total_cost,
                    "context_window": r.context_window,
                }
                for r in comparison.rows[:top]
            ],
        }, indent=2))
        return

    console.print()
    console.print(Rule("[bold cyan]Multi-Provider Cost Comparison[/bold cyan]"))

    summary_text = (
        f"[bold]Input Tokens:[/bold]   [yellow]{comparison.input_tokens:,}[/yellow]\n"
        f"[bold]Output Tokens:[/bold]  {comparison.output_tokens:,}\n"
        f"[bold]Cheapest:[/bold]       "
        f"[green]{comparison.cheapest.provider}/{comparison.cheapest.model}[/green] "
        f"— {format_cost(comparison.cheapest.total_cost)}\n"
        f"[bold]Most Expensive:[/bold] "
        f"[red]{comparison.most_expensive.provider}/{comparison.most_expensive.model}[/red] "
        f"— {format_cost(comparison.most_expensive.total_cost)}"
    )
    console.print(Panel(summary_text, title="[bold]Summary[/bold]",
                        border_style="cyan", padding=(1, 2)))

    console.print()
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("#",            justify="right", style="dim",    min_width=3)
    t.add_column("Provider",     style="cyan",                    min_width=12)
    t.add_column("Model",                                         min_width=28)
    t.add_column("Input Cost",   justify="right",                 min_width=12)
    t.add_column("Output Cost",  justify="right",                 min_width=12)
    t.add_column("Total Cost",   justify="right",                 min_width=12)
    t.add_column("Context Win.", justify="right",                 min_width=12)

    cheapest_total = comparison.cheapest.total_cost
    for i, r in enumerate(comparison.rows[:top], 1):
        multiple = (r.total_cost / cheapest_total) if cheapest_total > 0 else 1.0
        cost_color = "green" if i == 1 else "yellow" if multiple < 3 else "red"
        ctx_str = f"{r.context_window:,}" if r.context_window else "N/A"
        t.add_row(
            str(i), r.provider, r.model,
            format_cost(r.input_cost),
            format_cost(r.output_cost),
            f"[{cost_color}]{format_cost(r.total_cost)}[/{cost_color}]",
            ctx_str,
        )
    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# monitor-session
# ---------------------------------------------------------------------------

@cli.command("monitor-session")
@click.option("--prompt", "-p", default=None, metavar="TEXT",
              help="Prompt text to send (or use --file).")
@click.option("--file", "-f", "prompt_file", default=None, metavar="FILE",
              help="Read prompt from a file.")
@click.option("--model", "-m", default="claude-sonnet", show_default=True)
@click.option("--max-tokens", default=1024, show_default=True,
              help="Max output tokens for live API mode.")
@click.option("--simulate", is_flag=True,
              help="Run in simulation mode even when ANTHROPIC_API_KEY is set.")
@click.option("--sim-response", default=None, metavar="TEXT",
              help="Custom response text for simulation mode.")
def monitor_session(
    prompt: Optional[str],
    prompt_file: Optional[str],
    model: str,
    max_tokens: int,
    simulate: bool,
    sim_response: Optional[str],
):
    """Live token counter with real-time cost tracking.

    When ANTHROPIC_API_KEY is set, streams a real completion.
    Without the key (or with --simulate), runs in simulation mode.

    \b
    Examples:
      claude-toolkit monitor-session --prompt "Explain transformers"
      claude-toolkit monitor-session --file prompt.txt
      claude-toolkit monitor-session --prompt "Hello" --simulate
      claude-toolkit monitor-session --file prompt.txt --model claude-opus-4-6
    """
    if prompt_file:
        p = Path(prompt_file)
        if not p.exists():
            raise click.ClickException(f"File not found: {prompt_file}")
        prompt_text = p.read_text(encoding="utf-8")
    elif prompt:
        prompt_text = prompt
    else:
        console.print("[dim]Reading prompt from stdin (Ctrl-D when done)...[/dim]")
        prompt_text = sys.stdin.read()

    if not prompt_text.strip():
        raise click.ClickException("Prompt is empty.")

    console.print()
    console.print(Rule("[bold cyan]Live Token Monitor[/bold cyan]"))

    if simulate:
        response = sim_response or (
            "This is a simulated streaming response. "
            "The live monitor updates every 100 milliseconds showing input tokens, "
            "output tokens, tokens per second, elapsed time, and running cost estimate. "
            "In real API mode ANTHROPIC_API_KEY triggers an actual streaming completion."
        )
        stats = simulate_stream(prompt_text, response, model)
    else:
        stats = monitor_from_api(prompt_text, model, max_tokens)
        if stats is None:
            console.print("[yellow]Falling back to simulation mode.[/yellow]")
            response = sim_response or "Simulated response for demonstration purposes."
            stats = simulate_stream(prompt_text, response, model)

    from claude_toolkit.cost_estimator.estimator import estimate_cost
    final_cost = estimate_cost(stats.input_tokens, stats.output_tokens, model)
    console.print()
    console.print(
        f"  [bold]Final:[/bold] "
        f"[yellow]{stats.input_tokens:,}[/yellow] in + "
        f"[green]{stats.output_tokens:,}[/green] out = "
        f"[bold]{stats.total_tokens:,}[/bold] tokens  |  "
        f"Cost: [bold cyan]{format_cost(final_cost.total_cost)}[/bold cyan]  |  "
        f"{stats.elapsed_seconds:.1f}s @ {stats.tokens_per_second:.0f} tok/s"
    )
    console.print()


# ---------------------------------------------------------------------------
# serve (IDE API server)
# ---------------------------------------------------------------------------

@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Host to bind the API server.")
@click.option("--port", default=8765, show_default=True,
              help="Port to bind the API server.")
def serve(host: str, port: int):
    """Start the IDE integration JSON API server.

    Exposes a local HTTP API for VSCode, JetBrains, and other IDE clients
    to get live token counts, cost estimates, and optimization suggestions
    without running the CLI directly.

    \b
    Endpoints:
      GET  /health
      GET  /models
      POST /count          {text, model}
      POST /analyze        {text, model}
      POST /optimize       {text, model, aggressive}
      POST /rag-analyze    {text, threshold_tokens}
      POST /cache-detect   {prompts: [...], min_prefix_tokens, similarity_threshold}
      POST /budget-check   {session_cost, session_budget, daily_budget}

    \b
    Examples:
      claude-toolkit serve
      claude-toolkit serve --port 9000
      claude-toolkit serve --host 0.0.0.0 --port 8765
    """
    console.print()
    console.print(Rule("[bold cyan]Claude Toolkit IDE API Server[/bold cyan]"))
    console.print(f"\n  Listening on [bold cyan]http://{host}:{port}[/bold cyan]")
    console.print("  Press [bold]Ctrl-C[/bold] to stop.\n")
    console.print("  Endpoints:")
    for route in sorted([
        "GET  /health", "GET  /models",
        "POST /count", "POST /analyze", "POST /optimize",
        "POST /rag-analyze", "POST /cache-detect", "POST /budget-check",
    ]):
        console.print(f"    [cyan]{route}[/cyan]")
    console.print()

    server = ToolkitApiServer(host=host, port=port)
    try:
        server.start(background=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
