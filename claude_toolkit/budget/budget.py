"""
Cost budget alerts and daily spend tracking.

Configuration priority (highest → lowest):
  1. CLI --session-budget / --daily-budget flags
  2. Environment variables: CLAUDE_TOOLKIT_SESSION_BUDGET, CLAUDE_TOOLKIT_DAILY_BUDGET
  3. Config file: ~/.claude-toolkit/budget.json

Daily spend is accumulated in ~/.claude-toolkit/daily_spend.json and
entries older than 30 days are automatically pruned.

Usage
-----
    from claude_toolkit.budget.budget import load_budget_config, check_budget, record_spend

    cfg = load_budget_config(session_budget=5.00, daily_budget=20.00)
    alert = check_budget(session_cost=4.10, cfg=cfg, add_to_daily=True)
    print(alert.message)
"""

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from claude_toolkit.models import BudgetAlert, BudgetConfig


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CONFIG_DIR  = Path.home() / ".claude-toolkit"
_BUDGET_FILE = _CONFIG_DIR / "budget.json"
_SPEND_FILE  = _CONFIG_DIR / "daily_spend.json"


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def load_budget_config(
    session_budget: Optional[float] = None,
    daily_budget: Optional[float] = None,
    warn_at_pct: Optional[float] = None,
) -> BudgetConfig:
    """
    Load budget configuration, merging sources in priority order.

    Returns a BudgetConfig ready for use with check_budget().
    """
    cfg = BudgetConfig()

    # 1. Config file (lowest priority)
    if _BUDGET_FILE.exists():
        try:
            data = json.loads(_BUDGET_FILE.read_text(encoding="utf-8"))
            cfg.session_budget = data.get("session_budget")
            cfg.daily_budget   = data.get("daily_budget")
            cfg.warn_at_pct    = float(data.get("warn_at_pct", 80.0))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    # 2. Environment variables
    if env_s := os.environ.get("CLAUDE_TOOLKIT_SESSION_BUDGET"):
        try:
            cfg.session_budget = float(env_s)
        except ValueError:
            pass
    if env_d := os.environ.get("CLAUDE_TOOLKIT_DAILY_BUDGET"):
        try:
            cfg.daily_budget = float(env_d)
        except ValueError:
            pass
    if env_w := os.environ.get("CLAUDE_TOOLKIT_WARN_AT_PCT"):
        try:
            cfg.warn_at_pct = float(env_w)
        except ValueError:
            pass

    # 3. CLI overrides (highest priority)
    if session_budget is not None:
        cfg.session_budget = session_budget
    if daily_budget is not None:
        cfg.daily_budget = daily_budget
    if warn_at_pct is not None:
        cfg.warn_at_pct = warn_at_pct

    return cfg


def save_budget_config(cfg: BudgetConfig) -> None:
    """Persist a BudgetConfig to the config file."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "session_budget": cfg.session_budget,
        "daily_budget":   cfg.daily_budget,
        "warn_at_pct":    cfg.warn_at_pct,
    }
    _BUDGET_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Daily spend tracking
# ---------------------------------------------------------------------------

def _load_spend_data() -> dict:
    if not _SPEND_FILE.exists():
        return {}
    try:
        return json.loads(_SPEND_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_spend_data(data: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = str((datetime.today() - timedelta(days=30)).date())
    pruned = {k: v for k, v in data.items() if k >= cutoff}
    _SPEND_FILE.write_text(json.dumps(pruned, indent=2), encoding="utf-8")


def get_daily_spend(target_date: Optional[date] = None) -> float:
    """Return accumulated spend for *target_date* (defaults to today)."""
    key = str(target_date or date.today())
    return float(_load_spend_data().get(key, 0.0))


def record_spend(amount: float, target_date: Optional[date] = None) -> float:
    """
    Add *amount* to the daily spend tracker.

    Returns the new running total for the day.
    """
    key = str(target_date or date.today())
    data = _load_spend_data()
    data[key] = data.get(key, 0.0) + amount
    _save_spend_data(data)
    return float(data[key])


def get_spend_history(days: int = 7) -> dict:
    """Return a {date_str: spend_float} dict for the last *days* days."""
    data = _load_spend_data()
    result = {}
    for i in range(days):
        d = str((datetime.today() - timedelta(days=i)).date())
        result[d] = data.get(d, 0.0)
    return dict(sorted(result.items()))


# ---------------------------------------------------------------------------
# Budget evaluation
# ---------------------------------------------------------------------------

def check_budget(
    session_cost: float,
    cfg: Optional[BudgetConfig] = None,
    add_to_daily: bool = False,
) -> BudgetAlert:
    """
    Evaluate *session_cost* against configured session and daily budgets.

    Args:
        session_cost:  Total cost accrued in the current session (USD).
        cfg:           BudgetConfig to use; loaded from file if None.
        add_to_daily:  When True, records session_cost in the daily tracker
                       (call once per completed session).

    Returns:
        BudgetAlert with a severity level and human-readable message.
    """
    if cfg is None:
        cfg = load_budget_config()

    if add_to_daily:
        daily_cost = record_spend(session_cost)
    else:
        daily_cost = get_daily_spend() + session_cost

    # Compute percentages
    session_pct: Optional[float] = None
    if cfg.session_budget and cfg.session_budget > 0:
        session_pct = round(session_cost / cfg.session_budget * 100, 1)

    daily_pct: Optional[float] = None
    if cfg.daily_budget and cfg.daily_budget > 0:
        daily_pct = round(daily_cost / cfg.daily_budget * 100, 1)

    percentages = [p for p in [session_pct, daily_pct] if p is not None]
    worst_pct = max(percentages) if percentages else 0.0

    # No budgets configured
    if cfg.session_budget is None and cfg.daily_budget is None:
        return BudgetAlert(
            level="ok",
            message=(
                "No budget configured. "
                "Run `claude-toolkit set-budget --session 5.00 --daily 20.00` to enable alerts."
            ),
            session_cost=session_cost,
            session_budget=None,
            session_pct=None,
            daily_cost=daily_cost,
            daily_budget=None,
            daily_pct=None,
        )

    # Hard-exceeded checks
    if cfg.session_budget and session_cost > cfg.session_budget:
        level = "exceeded"
        message = (
            f"BUDGET EXCEEDED  Session cost ${session_cost:.4f} exceeds "
            f"session limit ${cfg.session_budget:.2f}. "
            "Consider switching to a smaller model or splitting the task."
        )
    elif cfg.daily_budget and daily_cost > cfg.daily_budget:
        level = "exceeded"
        message = (
            f"BUDGET EXCEEDED  Daily spend ${daily_cost:.4f} exceeds "
            f"daily limit ${cfg.daily_budget:.2f}. "
            "Stop new sessions or raise your budget."
        )
    elif worst_pct >= 100.0:
        level = "exceeded"
        message = f"Budget exceeded ({worst_pct:.0f}% of limit used)."
    elif worst_pct >= cfg.warn_at_pct:
        level = "warn"
        parts: list = []
        if session_pct is not None:
            parts.append(f"session at {session_pct:.0f}% (${session_cost:.4f} / ${cfg.session_budget:.2f})")
        if daily_pct is not None:
            parts.append(f"daily at {daily_pct:.0f}% (${daily_cost:.4f} / ${cfg.daily_budget:.2f})")
        message = (
            f"WARNING: {', '.join(parts)}. "
            "Reduce prompt size or switch to a cheaper model tier."
        )
    else:
        level = "ok"
        parts = []
        if session_pct is not None:
            parts.append(f"session {session_pct:.0f}%")
        if daily_pct is not None:
            parts.append(f"daily {daily_pct:.0f}%")
        message = f"Spend within budget ({', '.join(parts)})."

    return BudgetAlert(
        level=level,
        message=message,
        session_cost=session_cost,
        session_budget=cfg.session_budget,
        session_pct=session_pct,
        daily_cost=daily_cost,
        daily_budget=cfg.daily_budget,
        daily_pct=daily_pct,
    )
