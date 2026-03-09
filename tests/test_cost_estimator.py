"""Tests for claude_toolkit.cost_estimator.estimator"""

import pytest
from claude_toolkit.cost_estimator.estimator import (
    estimate_cost,
    format_cost,
    get_pricing,
    list_models,
    compare_providers,
    list_all_providers,
    MODEL_PRICING,
    PROVIDER_PRICING,
)
from claude_toolkit.models import CostEstimate, ProviderComparison


class TestEstimateCost:
    def test_returns_cost_estimate(self):
        result = estimate_cost(1000, 500, "claude-sonnet")
        assert isinstance(result, CostEstimate)

    def test_zero_tokens_zero_cost(self):
        result = estimate_cost(0, 0, "claude-sonnet")
        assert result.total_cost == 0.0
        assert result.input_cost == 0.0
        assert result.output_cost == 0.0

    def test_total_equals_input_plus_output(self):
        result = estimate_cost(1000, 500, "claude-sonnet")
        assert abs(result.total_cost - (result.input_cost + result.output_cost)) < 1e-10

    def test_opus_more_expensive_than_haiku(self):
        opus   = estimate_cost(1000, 500, "claude-opus")
        haiku  = estimate_cost(1000, 500, "claude-haiku")
        assert opus.total_cost > haiku.total_cost

    def test_more_tokens_higher_cost(self):
        small = estimate_cost(100,  50, "claude-sonnet")
        large = estimate_cost(1000, 500, "claude-sonnet")
        assert large.total_cost > small.total_cost

    def test_custom_pricing_overrides_table(self):
        custom = {"input_per_million": 1.0, "output_per_million": 2.0}
        result = estimate_cost(1_000_000, 1_000_000, "claude-sonnet",
                               custom_pricing=custom)
        assert abs(result.input_cost  - 1.0) < 0.001
        assert abs(result.output_cost - 2.0) < 0.001

    def test_model_alias_resolves(self):
        for alias in ("claude-sonnet", "claude-opus", "claude-haiku",
                      "claude-sonnet-4-6", "claude-opus-4-6"):
            r = estimate_cost(100, 50, alias)
            assert r.total_cost > 0


class TestFormatCost:
    def test_zero(self):
        assert format_cost(0) == "$0.00000"

    def test_large_cost_has_dollars(self):
        s = format_cost(1.50)
        assert "$" in s
        assert "1.5" in s

    def test_tiny_cost_uses_enough_decimals(self):
        s = format_cost(0.000001)
        assert "$" in s
        assert len(s) > 3

    def test_returns_string(self):
        assert isinstance(format_cost(0.123), str)


class TestGetPricing:
    def test_returns_dict_with_required_keys(self):
        p = get_pricing("claude-sonnet")
        assert "input_per_million"  in p
        assert "output_per_million" in p
        assert "context_window"     in p

    def test_unknown_model_falls_back_to_sonnet(self):
        p = get_pricing("totally-unknown-model-xyz")
        sonnet = get_pricing("claude-sonnet")
        assert p["input_per_million"] == sonnet["input_per_million"]


class TestListModels:
    def test_returns_dict(self):
        assert isinstance(list_models(), dict)

    def test_contains_claude_models(self):
        models = list_models()
        assert any("claude" in k for k in models)

    def test_is_copy(self):
        a = list_models()
        b = list_models()
        a["new_key"] = {}
        assert "new_key" not in b


class TestCompareProviders:
    def test_returns_provider_comparison(self):
        result = compare_providers(1000, 500)
        assert isinstance(result, ProviderComparison)

    def test_rows_sorted_cheapest_first(self):
        result = compare_providers(1000, 500)
        costs = [r.total_cost for r in result.rows]
        assert costs == sorted(costs)

    def test_cheapest_matches_first_row(self):
        result = compare_providers(1000, 500)
        assert result.cheapest.total_cost == result.rows[0].total_cost

    def test_most_expensive_matches_last_row(self):
        result = compare_providers(1000, 500)
        assert result.most_expensive.total_cost == result.rows[-1].total_cost

    def test_provider_filter(self):
        result = compare_providers(1000, 500, providers=["anthropic"])
        for r in result.rows:
            assert r.provider == "anthropic"

    def test_all_providers_covered(self):
        result = compare_providers(1000, 500)
        providers = {r.provider for r in result.rows}
        assert providers >= {"anthropic", "openai", "google", "bedrock"}

    def test_token_counts_propagated(self):
        result = compare_providers(1234, 567)
        assert result.input_tokens  == 1234
        assert result.output_tokens == 567


class TestProviderPricing:
    def test_all_providers_present(self):
        for p in ("anthropic", "openai", "google", "bedrock"):
            assert p in PROVIDER_PRICING

    def test_every_entry_has_required_keys(self):
        for provider, models in PROVIDER_PRICING.items():
            for model_id, pricing in models.items():
                assert "input_per_million"  in pricing, f"{provider}/{model_id} missing input_per_million"
                assert "output_per_million" in pricing, f"{provider}/{model_id} missing output_per_million"
                assert "context_window"     in pricing, f"{provider}/{model_id} missing context_window"

    def test_list_all_providers_is_copy(self):
        a = list_all_providers()
        b = list_all_providers()
        a["injected"] = {}
        assert "injected" not in b
