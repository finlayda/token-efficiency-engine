"""Shared data models for the Claude Token Toolkit."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CostEstimate:
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str


@dataclass
class OptimizationStrategy:
    name: str
    description: str
    tokens_saved: int
    applied: bool


@dataclass
class OptimizationResult:
    original_prompt: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    reduction_pct: float
    strategies_applied: List[OptimizationStrategy]
    suggestions: List[str]


@dataclass
class PromptAnalysis:
    prompt: str
    input_tokens: int
    estimated_output_tokens: int
    cost: CostEstimate
    section_breakdown: Dict[str, int]
    heatmap: Dict[str, float]
    suggestions: List[str]


@dataclass
class SessionEntry:
    prompt_id: int
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    model: str
    timestamp: Optional[str] = None


@dataclass
class SessionSummary:
    total_prompts: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost: float
    top_consumers: List[Dict[str, Any]]
    model_breakdown: Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# v2 — FinOps & Prompt Optimization Platform models
# ---------------------------------------------------------------------------

@dataclass
class CacheCandidate:
    """A prompt prefix or shared block that is a strong caching candidate."""
    prefix: str
    token_count: int
    frequency: int                  # number of prompts sharing this prefix
    prompt_ids: List[int]
    savings_pct: float
    estimated_tokens_saved: int


@dataclass
class CacheReport:
    """Session-level cache opportunity report."""
    candidates: List[CacheCandidate]
    total_prompts: int
    total_cacheable_tokens: int
    overall_savings_pct: float
    suggestion: str


@dataclass
class ContextBlock:
    """A repeated context block extracted from session prompts."""
    content: str
    token_count: int
    occurrences: int
    category: str           # "project", "auth", "database", "api", etc.


@dataclass
class ClaudeMdResult:
    """Result of auto-generating a CLAUDE.md shared context file."""
    blocks: List[ContextBlock]
    generated_content: str
    total_repeated_tokens: int
    estimated_savings_per_prompt: int
    output_path: Optional[str] = None


@dataclass
class BudgetConfig:
    """User-configured cost budget thresholds."""
    session_budget: Optional[float] = None
    daily_budget: Optional[float] = None
    warn_at_pct: float = 80.0       # warn when spend reaches this % of budget


@dataclass
class BudgetAlert:
    """Result of evaluating costs against configured budgets."""
    level: str                      # "ok" | "warn" | "critical" | "exceeded"
    message: str
    session_cost: float
    session_budget: Optional[float]
    session_pct: Optional[float]
    daily_cost: float
    daily_budget: Optional[float]
    daily_pct: Optional[float]


@dataclass
class RagContextBlock:
    """A large static context block detected in a prompt."""
    start_line: int
    end_line: int
    content: str
    token_count: int
    block_type: str                 # "document" | "code" | "data" | "freetext"
    retrieval_suggestion: str
    estimated_savings_pct: float


@dataclass
class RagAdvice:
    """Full RAG compression analysis for a prompt."""
    blocks: List[RagContextBlock]
    total_context_tokens: int
    total_prompt_tokens: int
    context_fraction: float         # % of prompt that is static context
    total_estimated_savings_pct: float
    recommendations: List[str]


@dataclass
class ExampleCluster:
    """A group of similar examples collapsed to one representative."""
    representative: str
    members: List[str]
    member_indices: List[int]
    similarity_score: float
    tokens_saved: int


@dataclass
class PruningResult:
    """Result of example clustering and pruning."""
    original_examples: List[str]
    kept_examples: List[str]
    clusters: List[ExampleCluster]
    original_tokens: int
    pruned_tokens: int
    tokens_saved: int
    reduction_pct: float


@dataclass
class StreamStats:
    """Live token/cost stats for a streaming session."""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "claude-sonnet"
    elapsed_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_second(self) -> float:
        return round(self.output_tokens / max(0.001, self.elapsed_seconds), 1)


@dataclass
class ProviderCostRow:
    """Single row in a multi-provider cost comparison."""
    provider: str
    model: str
    input_cost: float
    output_cost: float
    total_cost: float
    context_window: int


@dataclass
class ProviderComparison:
    """Multi-provider cost comparison for a single prompt."""
    input_tokens: int
    output_tokens: int
    rows: List[ProviderCostRow]
    cheapest: ProviderCostRow
    most_expensive: ProviderCostRow
