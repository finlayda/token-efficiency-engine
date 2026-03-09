"""Tests for claude_toolkit.budget.budget"""

import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pytest
from claude_toolkit.budget.budget import (
    check_budget,
    get_daily_spend,
    get_spend_history,
    load_budget_config,
    record_spend,
    save_budget_config,
)
from claude_toolkit.models import BudgetAlert, BudgetConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_budget_dir(tmp_path, monkeypatch):
    """Redirect all budget file I/O to a temporary directory."""
    import claude_toolkit.budget.budget as bmod
    monkeypatch.setattr(bmod, "_CONFIG_DIR",  tmp_path)
    monkeypatch.setattr(bmod, "_BUDGET_FILE", tmp_path / "budget.json")
    monkeypatch.setattr(bmod, "_SPEND_FILE",  tmp_path / "daily_spend.json")
    # Clear env vars that might influence tests
    monkeypatch.delenv("CLAUDE_TOOLKIT_SESSION_BUDGET", raising=False)
    monkeypatch.delenv("CLAUDE_TOOLKIT_DAILY_BUDGET",   raising=False)
    monkeypatch.delenv("CLAUDE_TOOLKIT_WARN_AT_PCT",    raising=False)
    yield tmp_path


# ---------------------------------------------------------------------------
# load_budget_config
# ---------------------------------------------------------------------------

class TestLoadBudgetConfig:
    def test_defaults_when_no_file(self):
        cfg = load_budget_config()
        assert cfg.session_budget is None
        assert cfg.daily_budget   is None
        assert cfg.warn_at_pct    == 80.0

    def test_cli_overrides_take_priority(self):
        cfg = load_budget_config(session_budget=3.0, daily_budget=10.0)
        assert cfg.session_budget == 3.0
        assert cfg.daily_budget   == 10.0

    def test_env_vars_loaded(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_TOOLKIT_SESSION_BUDGET", "2.50")
        monkeypatch.setenv("CLAUDE_TOOLKIT_DAILY_BUDGET",   "15.00")
        cfg = load_budget_config()
        assert cfg.session_budget == 2.50
        assert cfg.daily_budget   == 15.00

    def test_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_TOOLKIT_SESSION_BUDGET", "2.50")
        cfg = load_budget_config(session_budget=9.99)
        assert cfg.session_budget == 9.99

    def test_reads_from_file(self, isolated_budget_dir):
        data = {"session_budget": 5.0, "daily_budget": 20.0, "warn_at_pct": 70.0}
        (isolated_budget_dir / "budget.json").write_text(json.dumps(data))
        cfg = load_budget_config()
        assert cfg.session_budget == 5.0
        assert cfg.daily_budget   == 20.0
        assert cfg.warn_at_pct    == 70.0

    def test_invalid_json_in_file_uses_defaults(self, isolated_budget_dir):
        (isolated_budget_dir / "budget.json").write_text("not json!")
        cfg = load_budget_config()
        assert cfg.session_budget is None


# ---------------------------------------------------------------------------
# save_budget_config
# ---------------------------------------------------------------------------

class TestSaveBudgetConfig:
    def test_writes_json_file(self, isolated_budget_dir):
        cfg = BudgetConfig(session_budget=5.0, daily_budget=20.0, warn_at_pct=75.0)
        save_budget_config(cfg)
        data = json.loads((isolated_budget_dir / "budget.json").read_text())
        assert data["session_budget"] == 5.0
        assert data["daily_budget"]   == 20.0
        assert data["warn_at_pct"]    == 75.0

    def test_round_trips(self, isolated_budget_dir):
        cfg = BudgetConfig(session_budget=1.23, daily_budget=9.87)
        save_budget_config(cfg)
        loaded = load_budget_config()
        assert loaded.session_budget == pytest.approx(1.23)
        assert loaded.daily_budget   == pytest.approx(9.87)


# ---------------------------------------------------------------------------
# record_spend / get_daily_spend
# ---------------------------------------------------------------------------

class TestSpendTracking:
    def test_initial_spend_is_zero(self):
        assert get_daily_spend() == 0.0

    def test_record_adds_to_daily(self):
        total = record_spend(1.50)
        assert total == pytest.approx(1.50)
        assert get_daily_spend() == pytest.approx(1.50)

    def test_multiple_records_accumulate(self):
        record_spend(1.00)
        record_spend(0.50)
        assert get_daily_spend() == pytest.approx(1.50)

    def test_specific_date(self):
        yesterday = date.today() - timedelta(days=1)
        record_spend(2.00, target_date=yesterday)
        assert get_daily_spend(yesterday) == pytest.approx(2.00)
        assert get_daily_spend() == 0.0  # today unaffected


class TestGetSpendHistory:
    def test_returns_dict_with_correct_length(self):
        history = get_spend_history(7)
        assert len(history) == 7

    def test_dates_are_sorted(self):
        history = get_spend_history(5)
        dates = list(history.keys())
        assert dates == sorted(dates)

    def test_recorded_spend_appears_in_history(self):
        record_spend(3.14)
        history = get_spend_history(7)
        today = str(date.today())
        assert history[today] == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# check_budget
# ---------------------------------------------------------------------------

class TestCheckBudget:
    def test_returns_budget_alert(self):
        cfg = BudgetConfig(session_budget=5.0, daily_budget=20.0)
        alert = check_budget(1.0, cfg)
        assert isinstance(alert, BudgetAlert)

    def test_ok_when_well_within_budget(self):
        cfg = BudgetConfig(session_budget=10.0, daily_budget=50.0)
        alert = check_budget(1.0, cfg)
        assert alert.level == "ok"

    def test_warn_when_near_limit(self):
        cfg = BudgetConfig(session_budget=5.0, daily_budget=20.0, warn_at_pct=80.0)
        alert = check_budget(4.5, cfg)  # 90% of 5.0
        assert alert.level in ("warn", "exceeded")

    def test_exceeded_when_over_session_budget(self):
        cfg = BudgetConfig(session_budget=5.0)
        alert = check_budget(6.0, cfg)
        assert alert.level == "exceeded"

    def test_no_budget_configured_returns_ok(self):
        cfg = BudgetConfig()
        alert = check_budget(999.0, cfg)
        assert alert.level == "ok"

    def test_session_pct_calculated_correctly(self):
        cfg = BudgetConfig(session_budget=10.0)
        alert = check_budget(5.0, cfg)
        assert alert.session_pct == pytest.approx(50.0)

    def test_add_to_daily_records_spend(self):
        cfg = BudgetConfig(daily_budget=100.0)
        check_budget(3.0, cfg, add_to_daily=True)
        assert get_daily_spend() == pytest.approx(3.0)

    def test_message_is_non_empty_string(self):
        cfg = BudgetConfig(session_budget=5.0)
        alert = check_budget(1.0, cfg)
        assert isinstance(alert.message, str)
        assert len(alert.message) > 5
