"""
IDE Integration API Server.

A lightweight HTTP/JSON API that exposes core toolkit capabilities for
consumption by:
  - VSCode extensions
  - JetBrains plugins
  - Neovim / Vim LSP clients
  - Any HTTP consumer

Start the server:
    claude-toolkit serve                        # default 127.0.0.1:8765
    claude-toolkit serve --port 9000
    claude-toolkit serve --host 0.0.0.0 --port 8765

API endpoints
-------------
GET  /health           → {"status": "ok", "version": "2.0.0"}
GET  /models           → {"models": {...}}
POST /count            → token count for a text snippet
POST /analyze          → full prompt analysis (tokens + cost + heatmap)
POST /optimize         → apply optimization pipeline
POST /rag-analyze      → detect large context blocks
POST /cache-detect     → detect cacheable prefixes in a session
POST /budget-check     → evaluate cost against configured budgets

All POST endpoints accept a JSON body and return JSON.
CORS headers are included so browser-based tools can call the API directly.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from claude_toolkit.budget.budget import check_budget, load_budget_config
from claude_toolkit.cache_detector.detector import detect_cache_candidates
from claude_toolkit.cost_estimator.estimator import estimate_cost, format_cost, list_models
from claude_toolkit.prompt_optimizer.optimizer import optimize
from claude_toolkit.rag_advisor.advisor import analyze_rag_opportunities
from claude_toolkit.tokenizer.counter import count_tokens, estimate_output_tokens, section_heatmap

_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# Route handlers — each receives a parsed JSON body dict and returns a dict
# ---------------------------------------------------------------------------

def _handle_health(_body: Dict) -> Dict:
    return {
        "status":  "ok",
        "version": _VERSION,
        "endpoints": sorted(_ROUTES.keys()),
    }


def _handle_models(_body: Dict) -> Dict:
    return {"models": list_models()}


def _handle_count(body: Dict) -> Dict:
    text  = str(body.get("text", ""))
    model = str(body.get("model", "claude-sonnet"))
    input_tok  = count_tokens(text, model)
    output_tok = estimate_output_tokens(text, model)
    return {
        "model":                   model,
        "input_tokens":            input_tok,
        "estimated_output_tokens": output_tok,
        "total_estimated":         input_tok + output_tok,
    }


def _handle_analyze(body: Dict) -> Dict:
    text  = str(body.get("text", ""))
    model = str(body.get("model", "claude-sonnet"))

    input_tok  = count_tokens(text, model)
    output_tok = estimate_output_tokens(text, model)
    cost       = estimate_cost(input_tok, output_tok, model)
    heatmap    = section_heatmap(text, model)

    return {
        "model":                   model,
        "input_tokens":            input_tok,
        "estimated_output_tokens": output_tok,
        "section_heatmap":         heatmap,
        "cost": {
            "input":     cost.input_cost,
            "output":    cost.output_cost,
            "total":     cost.total_cost,
            "formatted": format_cost(cost.total_cost),
        },
    }


def _handle_optimize(body: Dict) -> Dict:
    text       = str(body.get("text", ""))
    model      = str(body.get("model", "claude-sonnet"))
    aggressive = bool(body.get("aggressive", False))

    result = optimize(text, model, aggressive=aggressive)

    return {
        "original_tokens":  result.original_tokens,
        "optimized_tokens": result.optimized_tokens,
        "tokens_saved":     result.tokens_saved,
        "reduction_pct":    result.reduction_pct,
        "optimized_text":   result.optimized_prompt,
        "suggestions":      result.suggestions,
        "strategies": [
            {
                "name":         s.name,
                "applied":      s.applied,
                "tokens_saved": s.tokens_saved,
            }
            for s in result.strategies_applied
        ],
    }


def _handle_rag_analyze(body: Dict) -> Dict:
    text      = str(body.get("text", ""))
    threshold = int(body.get("threshold_tokens", 300))

    advice = analyze_rag_opportunities(text, threshold)

    return {
        "total_prompt_tokens":         advice.total_prompt_tokens,
        "total_context_tokens":        advice.total_context_tokens,
        "context_fraction_pct":        advice.context_fraction,
        "total_estimated_savings_pct": advice.total_estimated_savings_pct,
        "blocks": [
            {
                "type":                  b.block_type,
                "token_count":           b.token_count,
                "estimated_savings_pct": b.estimated_savings_pct,
                "suggestion":            b.retrieval_suggestion,
            }
            for b in advice.blocks
        ],
        "recommendations": advice.recommendations,
    }


def _handle_cache_detect(body: Dict) -> Dict:
    raw: List = body.get("prompts", [])
    # Accept list of strings or list of {id, text} dicts
    pairs: list = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            pairs.append((i + 1, item))
        elif isinstance(item, dict):
            pairs.append((item.get("id", i + 1), str(item.get("text", ""))))

    report = detect_cache_candidates(
        pairs,
        min_prefix_tokens=int(body.get("min_prefix_tokens", 50)),
        similarity_threshold=float(body.get("similarity_threshold", 0.6)),
    )

    return {
        "total_prompts":          report.total_prompts,
        "total_cacheable_tokens": report.total_cacheable_tokens,
        "overall_savings_pct":    report.overall_savings_pct,
        "suggestion":             report.suggestion,
        "candidates": [
            {
                "prefix_preview":          c.prefix[:120],
                "token_count":             c.token_count,
                "frequency":               c.frequency,
                "prompt_ids":              c.prompt_ids,
                "estimated_tokens_saved":  c.estimated_tokens_saved,
                "savings_pct":             c.savings_pct,
            }
            for c in report.candidates
        ],
    }


def _handle_budget_check(body: Dict) -> Dict:
    session_cost = float(body.get("session_cost", 0.0))
    cfg = load_budget_config(
        session_budget=body.get("session_budget"),
        daily_budget=body.get("daily_budget"),
    )
    alert = check_budget(session_cost, cfg, add_to_daily=False)

    return {
        "level":          alert.level,
        "message":        alert.message,
        "session_cost":   alert.session_cost,
        "session_budget": alert.session_budget,
        "session_pct":    alert.session_pct,
        "daily_cost":     alert.daily_cost,
        "daily_budget":   alert.daily_budget,
        "daily_pct":      alert.daily_pct,
    }


_ROUTES: Dict[str, Any] = {
    "/health":        _handle_health,
    "/models":        _handle_models,
    "/count":         _handle_count,
    "/analyze":       _handle_analyze,
    "/optimize":      _handle_optimize,
    "/rag-analyze":   _handle_rag_analyze,
    "/cache-detect":  _handle_cache_detect,
    "/budget-check":  _handle_budget_check,
}


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class _ApiHandler(BaseHTTPRequestHandler):
    """Minimal HTTP/1.1 handler — no external web framework required."""

    def log_message(self, fmt, *args):  # silence default logging
        pass

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        handler = _ROUTES.get(path)
        if handler:
            try:
                self._send_json(200, handler({}))
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": str(exc)})
        else:
            self._send_json(404, {
                "error":     f"Unknown endpoint: {path}",
                "available": sorted(_ROUTES.keys()),
            })

    def do_POST(self):
        path = urlparse(self.path).path
        handler = _ROUTES.get(path)
        if not handler:
            self._send_json(404, {
                "error":     f"Unknown endpoint: {path}",
                "available": sorted(_ROUTES.keys()),
            })
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"Invalid JSON body: {exc}"})
            return

        try:
            result = handler(body)
            self._send_json(200, result)
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": str(exc)})


# ---------------------------------------------------------------------------
# Server class
# ---------------------------------------------------------------------------

class ToolkitApiServer:
    """
    Wrapper around a stdlib HTTPServer with optional background-thread support.

    Examples
    --------
    # Foreground (blocks):
        server = ToolkitApiServer(port=8765)
        server.start()

    # Background (non-blocking):
        server = ToolkitApiServer(port=8765)
        server.start(background=True)
        # ... do other work ...
        server.stop()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, background: bool = False) -> None:
        self._server = HTTPServer((self.host, self.port), _ApiHandler)
        if background:
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True, name="toolkit-api"
            )
            self._thread.start()
        else:
            self._server.serve_forever()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
