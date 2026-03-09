"""Tests for claude_toolkit.api_server.server"""

import json
import time
import urllib.request
import urllib.error
import pytest
from claude_toolkit.api_server.server import ToolkitApiServer

PORT = 19876   # Use a non-standard port to avoid conflicts


@pytest.fixture(scope="module")
def api_server():
    """Start the API server once for all tests in this module."""
    srv = ToolkitApiServer(host="127.0.0.1", port=PORT)
    srv.start(background=True)
    time.sleep(0.3)   # Give the server time to bind
    yield srv
    srv.stop()


def _get(path: str) -> dict:
    resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5)
    return json.loads(resp.read())


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=5)
    return json.loads(resp.read())


class TestHealthEndpoint:
    def test_returns_ok(self, api_server):
        data = _get("/health")
        assert data["status"] == "ok"

    def test_returns_version(self, api_server):
        data = _get("/health")
        assert "version" in data
        assert data["version"] == "2.0.0"

    def test_lists_endpoints(self, api_server):
        data = _get("/health")
        assert "endpoints" in data
        assert "/analyze" in data["endpoints"]


class TestModelsEndpoint:
    def test_returns_models_dict(self, api_server):
        data = _get("/models")
        assert "models" in data
        assert isinstance(data["models"], dict)

    def test_contains_claude_models(self, api_server):
        data = _get("/models")
        assert any("claude" in k for k in data["models"])


class TestCountEndpoint:
    def test_returns_token_count(self, api_server):
        data = _post("/count", {"text": "Hello world", "model": "claude-sonnet"})
        assert "input_tokens" in data
        assert data["input_tokens"] >= 1

    def test_returns_output_estimate(self, api_server):
        data = _post("/count", {"text": "Write a function.", "model": "claude-sonnet"})
        assert "estimated_output_tokens" in data
        assert data["estimated_output_tokens"] >= 1

    def test_empty_text(self, api_server):
        data = _post("/count", {"text": "", "model": "claude-sonnet"})
        assert data["input_tokens"] == 0

    def test_model_field_returned(self, api_server):
        data = _post("/count", {"text": "test", "model": "claude-haiku"})
        assert data["model"] == "claude-haiku"


class TestAnalyzeEndpoint:
    def test_returns_analysis(self, api_server):
        data = _post("/analyze", {"text": "Explain machine learning.", "model": "claude-sonnet"})
        assert "input_tokens"    in data
        assert "cost"            in data
        assert "section_heatmap" in data

    def test_cost_structure(self, api_server):
        data = _post("/analyze", {"text": "Write code.", "model": "claude-sonnet"})
        cost = data["cost"]
        assert "input"     in cost
        assert "output"    in cost
        assert "total"     in cost
        assert "formatted" in cost

    def test_total_cost_positive(self, api_server):
        data = _post("/analyze", {"text": "A non-trivial prompt.", "model": "claude-sonnet"})
        assert data["cost"]["total"] > 0


class TestOptimizeEndpoint:
    def test_returns_optimization(self, api_server):
        text = "Could you please help me by kindly writing a function?"
        data = _post("/optimize", {"text": text, "model": "claude-sonnet"})
        assert "original_tokens"  in data
        assert "optimized_tokens" in data
        assert "tokens_saved"     in data

    def test_optimized_tokens_lte_original(self, api_server):
        text = "Please please please write some code for me if you could."
        data = _post("/optimize", {"text": text})
        assert data["optimized_tokens"] <= data["original_tokens"]

    def test_suggestions_is_list(self, api_server):
        data = _post("/optimize", {"text": "Write a function."})
        assert isinstance(data["suggestions"], list)


class TestRagAnalyzeEndpoint:
    def test_returns_rag_advice(self, api_server):
        data = _post("/rag-analyze", {
            "text": "Analyze this: " + "word " * 200,
            "threshold_tokens": 50,
        })
        assert "total_prompt_tokens"         in data
        assert "total_context_tokens"        in data
        assert "total_estimated_savings_pct" in data
        assert "blocks"                      in data
        assert "recommendations"             in data

    def test_short_prompt_no_blocks(self, api_server):
        data = _post("/rag-analyze", {
            "text": "Short prompt.",
            "threshold_tokens": 5000,
        })
        assert data["blocks"] == []


class TestCacheDetectEndpoint:
    def test_returns_cache_report(self, api_server):
        prompts = [
            {"id": 1, "text": "You are an engineer. Write login endpoint."},
            {"id": 2, "text": "You are an engineer. Write logout endpoint."},
            {"id": 3, "text": "You are an engineer. Add JWT refresh."},
        ]
        data = _post("/cache-detect", {
            "prompts": prompts,
            "min_prefix_tokens": 3,
        })
        assert "total_prompts"    in data
        assert "candidates"       in data
        assert "suggestion"       in data

    def test_accepts_string_list(self, api_server):
        data = _post("/cache-detect", {
            "prompts": ["Hello world", "Hello there", "Goodbye"],
            "min_prefix_tokens": 1,
        })
        assert data["total_prompts"] == 3


class TestBudgetCheckEndpoint:
    def test_returns_alert(self, api_server):
        data = _post("/budget-check", {
            "session_cost": 1.0,
            "session_budget": 5.0,
            "daily_budget": 20.0,
        })
        assert "level"         in data
        assert "message"       in data
        assert "session_cost"  in data

    def test_exceeded_level(self, api_server):
        data = _post("/budget-check", {
            "session_cost": 10.0,
            "session_budget": 5.0,
        })
        assert data["level"] == "exceeded"

    def test_ok_level(self, api_server):
        data = _post("/budget-check", {
            "session_cost": 1.0,
            "session_budget": 100.0,
        })
        assert data["level"] == "ok"


class TestUnknownEndpoint:
    def test_404_for_unknown_route(self, api_server):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get("/nonexistent")
        assert exc_info.value.code == 404
