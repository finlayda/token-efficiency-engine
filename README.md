# Claude Token Toolkit

> LLM FinOps & Prompt Optimization Platform for Claude Code workflows.

Analyse token usage, reduce prompt costs, detect caching opportunities, and monitor live spend — all from the CLI or a local JSON API.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [analyze-prompt](#analyze-prompt)
  - [optimize-prompt](#optimize-prompt)
  - [analyze-session](#analyze-session)
  - [compare](#compare)
  - [detect-cache](#detect-cache)
  - [extract-context](#extract-context)
  - [set-budget / check-budget](#set-budget--check-budget)
  - [rag-analyze](#rag-analyze)
  - [prune-examples](#prune-examples)
  - [compare-providers](#compare-providers)
  - [monitor-session](#monitor-session)
  - [serve](#serve)
  - [list-models](#list-models)
- [IDE Integration API](#ide-integration-api)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)

---

## Installation

```bash
git clone https://github.com/finlayda/token-efficiency-engine.git
cd token-efficiency-engine
pip install -e .

# Optional: enable live API streaming in monitor-session
pip install -e ".[api]"
```

**Requirements:** Python 3.9+, no external API key needed for most commands.

---

## Quick Start

```bash
# Analyse a prompt
claude-toolkit analyze-prompt examples/prompts/verbose_prompt.txt

# Optimize it
claude-toolkit optimize-prompt examples/prompts/verbose_prompt.txt --show-diff

# Analyse a session log
claude-toolkit analyze-session examples/logs/claude_session.json

# Compare cost across all providers
claude-toolkit compare-providers examples/prompts/verbose_prompt.txt

# Set a budget and check spend
claude-toolkit set-budget --session 5.00 --daily 20.00
claude-toolkit check-budget 1.23

# Start the IDE API server
claude-toolkit serve
```

---

## Commands

### analyze-prompt

Counts tokens, estimates output tokens, calculates cost, and generates a section heatmap and optimization suggestions.

```bash
claude-toolkit analyze-prompt prompt.txt
claude-toolkit analyze-prompt prompt.txt --model claude-opus-4-6
claude-toolkit analyze-prompt prompt.txt --heatmap
echo "your prompt" | claude-toolkit analyze-prompt -
claude-toolkit analyze-prompt prompt.txt --json
```

**Output**

```
──────────────────────── Prompt Analysis ─────────────────────────
 Model                          claude-sonnet
 Input Tokens                   581
 Est. Output Tokens             407
 Total Tokens (estimated)       988
 ──────────────────────────────────────────────────────────────────
 Input Rate                     $3.00 / M tokens
 Output Rate                    $15.00 / M tokens
 Total Estimated Cost           $0.007845
```

---

### optimize-prompt

Applies a 7-stage pipeline to reduce token count while preserving meaning.

```bash
claude-toolkit optimize-prompt prompt.txt
claude-toolkit optimize-prompt prompt.txt --aggressive
claude-toolkit optimize-prompt prompt.txt --show-diff
claude-toolkit optimize-prompt prompt.txt -o optimized.txt
```

Strategies applied:

1. Remove social filler
2. Compress verbose phrases
3. Deduplicate sentences
4. Deduplicate list items
5. Remove excessive formatting
6. *(aggressive)* Convert paragraphs to bullets
7. *(aggressive)* Compress examples

---

### analyze-session

Parses a JSON session log and reports aggregate token usage, cost, model breakdown, and top token consumers.

```bash
claude-toolkit analyze-session session.json
claude-toolkit analyze-session session.json --top 10
claude-toolkit analyze-session session.json --json
```

**Session log format** — accepts any of the following shapes:

```json
[
  { "prompt": "...", "response": "...", "model": "claude-sonnet" },
  { "input": "...",  "output": "...",  "model": "claude-haiku",
    "input_tokens": 120, "output_tokens": 80 }
]
```

---

### compare

Side-by-side token and cost comparison of two prompt files.

```bash
claude-toolkit compare original.txt optimized.txt
claude-toolkit compare v1.txt v2.txt --model claude-opus-4-6
```

---

### detect-cache

Scans a session log for repeated prompt prefixes or shared opening blocks that could be moved to a stable system prompt and cached via [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).

```bash
claude-toolkit detect-cache session.json
claude-toolkit detect-cache session.json --min-tokens 100
claude-toolkit detect-cache session.json --json
```

**Example output**

```
 Total Prompts:      10
 Cacheable Tokens:   1,540
 Est. Savings:       71.0%

 Suggestion:
 Move the 220-token shared prefix (used in 7 prompts) to a stable
 system prompt and enable prompt caching (cache_control: ephemeral).
```

---

### extract-context

Mines a session log for repeated context phrases and generates a `CLAUDE.md` file that Claude Code automatically injects into every session, eliminating per-prompt repetition.

```bash
claude-toolkit extract-context session.json
claude-toolkit extract-context session.json --project MyApp -o CLAUDE.md
claude-toolkit extract-context session.json --dry-run
```

**Example output**

```
 Blocks detected: 12   Est. savings per prompt: 245 tokens

 Generated CLAUDE.md:
 ## Api
 - Should we use synchronous REST calls, gRPC, or GraphQL?

 ## Stack
 - The project uses FastAPI and PostgreSQL.
```

---

### set-budget / check-budget

Configure session and daily cost limits. Warnings fire when spend approaches the threshold (default 80%).

```bash
# Configure limits
claude-toolkit set-budget --session 5.00 --daily 20.00
claude-toolkit set-budget --warn-at 70

# Evaluate a cost amount
claude-toolkit check-budget 4.10
claude-toolkit check-budget 1.23 --record   # adds to today's daily ledger
claude-toolkit check-budget 0 --history     # show 7-day spend history
```

Settings are saved to `~/.claude-toolkit/budget.json`.
Environment variables: `CLAUDE_TOOLKIT_SESSION_BUDGET`, `CLAUDE_TOOLKIT_DAILY_BUDGET`, `CLAUDE_TOOLKIT_WARN_AT_PCT`.

**Alert levels:** `ok` · `warn` · `exceeded`

---

### rag-analyze

Detects large static context blocks embedded in a prompt and recommends replacing them with retrieval-augmented generation (RAG) to reduce token usage by ~70%.

```bash
claude-toolkit rag-analyze prompt.txt
claude-toolkit rag-analyze prompt.txt --threshold 200
cat big_prompt.txt | claude-toolkit rag-analyze -
```

**Example output**

```
 Total Prompt Tokens:    3,200
 Large Context Tokens:   2,900  (91% of prompt)
 Est. RAG Savings:       64%

 Block #1  type=document  tokens=2,900
 Suggestion: Embed this document in a vector store and retrieve
 only the top-k semantically relevant chunks at query time.
```

---

### prune-examples

Clusters similar in-prompt examples using TF-IDF cosine similarity and keeps only one representative per cluster, saving tokens without losing coverage.

```bash
claude-toolkit prune-examples prompt.txt
claude-toolkit prune-examples prompt.txt --threshold 0.70
```

**Example output**

```
 Examples found:   5
 Clusters formed:  2
 Examples kept:    2
 Tokens saved:     380  (28.4%)
```

---

### compare-providers

Counts tokens in a prompt and prices the call against 16 models across Anthropic, OpenAI, Google, and AWS Bedrock — sorted cheapest first.

```bash
claude-toolkit compare-providers prompt.txt
claude-toolkit compare-providers prompt.txt --providers anthropic,openai
claude-toolkit compare-providers prompt.txt --top 5
```

**Example output**

```
 Input: 581 tokens  |  Output: 871 tokens
 Cheapest:       google/gemini-1.5-flash   — $0.000305
 Most Expensive: anthropic/claude-opus-4-6 — $0.0740

 #   Provider    Model                  Total Cost
 1   google      gemini-1.5-flash       $0.000305
 2   openai      gpt-4o-mini            $0.000610
 3   anthropic   claude-haiku-4-5       $0.003949
 4   anthropic   claude-sonnet-4-6      $0.014808
 5   anthropic   claude-opus-4-6        $0.074040
```

---

### monitor-session

Displays a live Rich dashboard of token count and cost as a response streams in. Runs in simulation mode by default; connects to the real Anthropic API when `ANTHROPIC_API_KEY` is set.

```bash
claude-toolkit monitor-session --prompt "Explain transformers"
claude-toolkit monitor-session --file prompt.txt
claude-toolkit monitor-session --prompt "Hello" --simulate
export ANTHROPIC_API_KEY=sk-ant-...
claude-toolkit monitor-session --file prompt.txt --model claude-sonnet-4-6
```

**Live display**

```
 ┌── Live Token Monitor ──────────────┐
 │ Model         claude-sonnet        │
 │ Input tokens  8                    │
 │ Output tokens 142      ← live      │
 │ Total tokens  150                  │
 │ Elapsed       2.1s                 │
 │ Tokens / sec  67.6                 │
 │ Live cost     $0.002154            │
 └────────────────────────────────────┘
```

---

### serve

Starts a local HTTP JSON API server for IDE extension integration.

```bash
claude-toolkit serve                        # 127.0.0.1:8765
claude-toolkit serve --port 9000
claude-toolkit serve --host 0.0.0.0 --port 8765
```

See [IDE Integration API](#ide-integration-api) for endpoint details.

---

### list-models

Displays the Anthropic pricing table.

```bash
claude-toolkit list-models
claude-toolkit list-models --json
```

---

## IDE Integration API

The `serve` command exposes a zero-dependency HTTP/JSON API that any IDE plugin can consume for live token counting and cost estimates.

| Method | Endpoint | Body |
|--------|----------|------|
| GET | `/health` | — |
| GET | `/models` | — |
| POST | `/count` | `{ "text", "model" }` |
| POST | `/analyze` | `{ "text", "model" }` |
| POST | `/optimize` | `{ "text", "model", "aggressive" }` |
| POST | `/rag-analyze` | `{ "text", "threshold_tokens" }` |
| POST | `/cache-detect` | `{ "prompts": [...], "min_prefix_tokens" }` |
| POST | `/budget-check` | `{ "session_cost", "session_budget", "daily_budget" }` |

**VSCode extension snippet**

```typescript
const res = await fetch('http://127.0.0.1:8765/count', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: editor.document.getText(), model: 'claude-sonnet' }),
});
const { input_tokens } = await res.json();
statusBar.text = `$(pulse) ${input_tokens} tokens`;
```

All endpoints return JSON and include CORS headers.

---

## Project Structure

```
claude_toolkit/
├── models.py                  Shared dataclasses
├── tokenizer/counter.py       tiktoken-based token counting
├── cost_estimator/estimator.py  Anthropic + multi-provider pricing
├── prompt_optimizer/optimizer.py  7-stage optimization pipeline
├── session_analyzer/analyzer.py  JSON session log parser
├── heatmap/heatmap.py         Section token density rendering
├── cache_detector/detector.py    Prompt prefix cache detection
├── context_extractor/extractor.py  CLAUDE.md generator
├── budget/budget.py           Cost budget alerts & spend ledger
├── rag_advisor/advisor.py     Large context block detection
├── example_pruner/pruner.py   TF-IDF example clustering
├── monitor/tracker.py         Live Rich token monitor
├── api_server/server.py       IDE integration HTTP API
└── cli_interface/cli.py       CLI entry point (14 commands)

tests/                         184 tests, all passing
examples/
├── prompts/verbose_prompt.txt
├── prompts/focused_prompt.txt
└── logs/claude_session.json
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

```
184 passed in 1.45s
```

---

## Provider Pricing Coverage

| Provider | Models |
|----------|--------|
| Anthropic | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 |
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o3-mini |
| Google | gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite |
| AWS Bedrock | bedrock-claude-sonnet-4-6, bedrock-claude-haiku-4-5, bedrock-titan-text-premier, bedrock-llama-3-70b |

All prices are sourced from each provider's published rates (USD per million tokens) and can be overridden with `--json` output piped into custom tooling.
