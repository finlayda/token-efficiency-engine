"""
Claude Token Toolkit — Streamlit UI
=====================================
Run with:
    cd token-efficiency-engine
    streamlit run ui/app.py
"""

import json
import tempfile
import os
import sys

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Make sure the package is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from claude_toolkit.tokenizer.counter import count_tokens, estimate_output_tokens, section_heatmap
from claude_toolkit.cost_estimator.estimator import (
    estimate_cost, format_cost, compare_providers, list_all_providers,
)
from claude_toolkit.prompt_optimizer.optimizer import optimize
from claude_toolkit.session_analyzer.analyzer import analyze_session_data, parse_session_entries
from claude_toolkit.cache_detector.detector import detect_cache_candidates
from claude_toolkit.context_extractor.extractor import generate_claude_md
from claude_toolkit.budget.budget import (
    load_budget_config, check_budget, get_spend_history, record_spend,
)
from claude_toolkit.rag_advisor.advisor import analyze_rag_opportunities
from claude_toolkit.example_pruner.pruner import prune_examples

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Claude Token Toolkit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = [
    "Prompt Analyzer",
    "Prompt Optimizer",
    "Session Analyzer",
    "Cache Detector",
    "Context Extractor",
    "RAG Advisor",
    "Example Pruner",
    "Provider Comparison",
    "Budget Tracker",
]

with st.sidebar:
    st.title("⚡ Claude Token Toolkit")
    st.caption("LLM FinOps & Prompt Optimization")
    st.divider()
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("v2.0.0 · [GitHub](https://github.com/finlayda/token-efficiency-engine)")

MODELS = [
    "claude-sonnet",
    "claude-haiku",
    "claude-opus",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-6",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gauge(value: float, max_val: float, title: str, suffix: str = "") -> go.Figure:
    """A simple gauge chart using Plotly."""
    pct = min(value / max_val * 100, 100) if max_val else 0
    color = "#22c55e" if pct < 60 else "#f59e0b" if pct < 85 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 28}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar":  {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, max_val * 0.6],  "color": "#14532d"},
                {"range": [max_val * 0.6, max_val * 0.85], "color": "#713f12"},
                {"range": [max_val * 0.85, max_val],       "color": "#7f1d1d"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=0, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    return fig


def _bar_chart(data: dict, title: str, xlabel: str, ylabel: str,
               color: str = "#6366f1") -> go.Figure:
    df = pd.DataFrame(list(data.items()), columns=[xlabel, ylabel])
    fig = px.bar(df, x=xlabel, y=ylabel, title=title, color_discrete_sequence=[color])
    fig.update_layout(
        height=300, margin=dict(t=40, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#334155"),
    )
    return fig


# ===========================================================================
# PAGE: Prompt Analyzer
# ===========================================================================

if page == "Prompt Analyzer":
    st.title("📊 Prompt Analyzer")
    st.caption("Count tokens, estimate output, and visualise section distribution.")

    col_input, col_opts = st.columns([3, 1])
    with col_input:
        prompt = st.text_area("Prompt", height=260,
                              placeholder="Paste your prompt here…")
    with col_opts:
        model = st.selectbox("Model", MODELS, key="ap_model")
        show_heatmap = st.toggle("Per-section heatmap", value=True)
        run = st.button("Analyse", type="primary", use_container_width=True)

    if run and prompt.strip():
        input_tok  = count_tokens(prompt, model)
        output_tok = estimate_output_tokens(prompt, model)
        cost       = estimate_cost(input_tok, output_tok, model)
        heatmap    = section_heatmap(prompt, model)

        st.divider()

        # Metric row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input Tokens",          f"{input_tok:,}")
        c2.metric("Est. Output Tokens",    f"{output_tok:,}")
        c3.metric("Total Tokens",          f"{input_tok + output_tok:,}")
        c4.metric("Estimated Cost",        format_cost(cost.total_cost))

        # Cost breakdown
        st.divider()
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Cost Breakdown")
            cost_df = pd.DataFrame({
                "Type":  ["Input", "Output"],
                "Cost ($)": [cost.input_cost, cost.output_cost],
            })
            fig = px.pie(cost_df, names="Type", values="Cost ($)",
                         color_discrete_sequence=["#6366f1", "#22c55e"],
                         hole=0.5)
            fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0", margin=dict(t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with cc2:
            if show_heatmap and heatmap:
                st.subheader("Section Token Distribution")
                st.plotly_chart(
                    _bar_chart(heatmap, "", "Section", "% of Tokens"),
                    use_container_width=True,
                )

        # Suggestions
        opt = optimize(prompt, model)
        if opt.suggestions:
            st.divider()
            st.subheader("💡 Suggestions")
            for s in opt.suggestions:
                st.info(s)

    elif run:
        st.warning("Please enter a prompt.")


# ===========================================================================
# PAGE: Prompt Optimizer
# ===========================================================================

elif page == "Prompt Optimizer":
    st.title("✂️ Prompt Optimizer")
    st.caption("Automatically reduce token count using a 7-stage pipeline.")

    col_l, col_r = st.columns([3, 1])
    with col_l:
        prompt = st.text_area("Original Prompt", height=260,
                              placeholder="Paste your prompt here…")
    with col_r:
        model      = st.selectbox("Model", MODELS, key="op_model")
        aggressive = st.toggle("Aggressive mode",
                               help="Also restructures paragraphs and compresses examples.")
        run = st.button("Optimize", type="primary", use_container_width=True)

    if run and prompt.strip():
        result     = optimize(prompt, model, aggressive=aggressive)
        orig_out   = estimate_output_tokens(prompt, model)
        opt_out    = estimate_output_tokens(result.optimized_prompt, model)
        orig_cost  = estimate_cost(result.original_tokens,  orig_out, model)
        opt_cost   = estimate_cost(result.optimized_tokens, opt_out,  model)
        saved_cost = orig_cost.total_cost - opt_cost.total_cost

        st.divider()

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Tokens",  f"{result.original_tokens:,}")
        c2.metric("Optimized Tokens", f"{result.optimized_tokens:,}",
                  delta=f"-{result.tokens_saved:,}", delta_color="inverse")
        c3.metric("Reduction",        f"{result.reduction_pct:.1f}%")
        c4.metric("Cost Saved",       format_cost(saved_cost))

        # Strategies table
        st.divider()
        st.subheader("Strategies Applied")
        strats = [
            {
                "Strategy":     s.name,
                "Applied":      "✓" if s.applied else "–",
                "Tokens Saved": s.tokens_saved,
                "Description":  s.description,
            }
            for s in result.strategies_applied
        ]
        st.dataframe(pd.DataFrame(strats), use_container_width=True, hide_index=True)

        # Side-by-side diff
        st.divider()
        st.subheader("Before / After")
        left, right = st.columns(2)
        with left:
            st.caption(f"**Original** — {result.original_tokens:,} tokens")
            st.code(result.original_prompt, language=None)
        with right:
            st.caption(f"**Optimized** — {result.optimized_tokens:,} tokens")
            st.code(result.optimized_prompt, language=None)

        # Download
        st.download_button(
            "⬇ Download optimized prompt",
            data=result.optimized_prompt,
            file_name="optimized_prompt.txt",
            mime="text/plain",
        )

        # Suggestions
        if result.suggestions:
            st.divider()
            st.subheader("💡 Further Suggestions")
            for s in result.suggestions:
                st.info(s)

    elif run:
        st.warning("Please enter a prompt.")


# ===========================================================================
# PAGE: Session Analyzer
# ===========================================================================

elif page == "Session Analyzer":
    st.title("📋 Session Analyzer")
    st.caption("Upload a Claude session log to see aggregate token usage and cost.")

    uploaded = st.file_uploader("Session log (JSON)", type=["json"])
    model    = st.selectbox("Default model for cost estimates", MODELS, key="sa_model")

    if uploaded:
        try:
            data    = json.load(uploaded)
            summary = analyze_session_data(data)
        except Exception as exc:
            st.error(f"Failed to parse log: {exc}")
            st.stop()

        st.divider()

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Prompts",       f"{summary.total_prompts:,}")
        c2.metric("Input Tokens",  f"{summary.total_input_tokens:,}")
        c3.metric("Output Tokens", f"{summary.total_output_tokens:,}")
        c4.metric("Total Tokens",  f"{summary.total_tokens:,}")
        c5.metric("Total Cost",    format_cost(summary.total_cost))

        # Model breakdown
        if summary.model_breakdown:
            st.divider()
            st.subheader("Model Breakdown")
            mb_rows = [
                {
                    "Model":         m,
                    "Prompts":       v["prompts"],
                    "Input Tokens":  v["input_tokens"],
                    "Output Tokens": v["output_tokens"],
                    "Cost":          format_cost(v["cost"]),
                }
                for m, v in summary.model_breakdown.items()
            ]
            st.dataframe(pd.DataFrame(mb_rows), use_container_width=True, hide_index=True)

        # Top consumers chart
        st.divider()
        st.subheader("Top Token Consumers")
        top_n = st.slider("Show top N", 3, min(20, summary.total_prompts), 10)
        top   = summary.top_consumers[:top_n]

        df_top = pd.DataFrame([
            {
                "Prompt": f"#{e['prompt_id']} {e['prompt_preview'][:40]}",
                "Input":  e["input_tokens"],
                "Output": e["output_tokens"],
                "Cost":   e["cost"],
            }
            for e in top
        ])

        fig = px.bar(df_top, x="Prompt", y=["Input", "Output"],
                     barmode="stack", title="Token Usage per Prompt",
                     color_discrete_sequence=["#6366f1", "#22c55e"])
        fig.update_layout(
            height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", xaxis_tickangle=-30,
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#334155"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_top, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE: Cache Detector
# ===========================================================================

elif page == "Cache Detector":
    st.title("🗄️ Cache Detector")
    st.caption("Find repeated prompt prefixes that could be cached to cut costs.")

    uploaded   = st.file_uploader("Session log (JSON)", type=["json"])
    min_tokens = st.slider("Min prefix tokens", 10, 200, 50)
    threshold  = st.slider("Similarity threshold", 0.3, 1.0, 0.6, step=0.05)

    if uploaded:
        try:
            data    = json.load(uploaded)
            entries = parse_session_entries(data)
            pairs   = [(e.prompt_id, e.prompt) for e in entries if e.prompt.strip()]
        except Exception as exc:
            st.error(f"Failed to parse log: {exc}")
            st.stop()

        report = detect_cache_candidates(pairs, min_tokens, threshold)
        st.divider()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Prompts",     report.total_prompts)
        c2.metric("Cacheable Tokens",  f"{report.total_cacheable_tokens:,}")
        c3.metric("Est. Savings",      f"{report.overall_savings_pct:.1f}%")

        if report.candidates:
            st.divider()
            st.subheader("Cache Candidates")

            rows = [
                {
                    "Prefix Preview":       c.prefix[:80] + ("…" if len(c.prefix) > 80 else ""),
                    "Tokens":               c.token_count,
                    "Frequency":            c.frequency,
                    "Est. Tokens Saved":    c.estimated_tokens_saved,
                    "Savings %":            f"{c.savings_pct:.1f}%",
                }
                for c in report.candidates
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Savings bar chart
            fig = px.bar(
                pd.DataFrame({"Candidate": [f"#{i+1}" for i in range(len(report.candidates))],
                              "Tokens Saved": [c.estimated_tokens_saved for c in report.candidates]}),
                x="Candidate", y="Tokens Saved",
                color_discrete_sequence=["#6366f1"],
                title="Estimated Tokens Saved per Candidate",
            )
            fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                              xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#334155"))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.info(report.suggestion)


# ===========================================================================
# PAGE: Context Extractor
# ===========================================================================

elif page == "Context Extractor":
    st.title("📝 Context Extractor")
    st.caption("Mine repeated context from a session and generate a CLAUDE.md file.")

    uploaded       = st.file_uploader("Session log (JSON)", type=["json"])
    project_name   = st.text_input("Project name", value="Project")
    min_occ        = st.slider("Min occurrences", 2, 5, 2)

    if uploaded:
        try:
            data    = json.load(uploaded)
            entries = parse_session_entries(data)
            prompts = [e.prompt for e in entries if e.prompt.strip()]
        except Exception as exc:
            st.error(f"Failed to parse log: {exc}")
            st.stop()

        result = generate_claude_md(prompts, project_name=project_name,
                                    min_occurrences=min_occ)
        st.divider()

        c1, c2, c3 = st.columns(3)
        c1.metric("Blocks Detected",          len(result.blocks))
        c2.metric("Total Repeated Tokens",    f"{result.total_repeated_tokens:,}")
        c3.metric("Savings / Prompt",         f"{result.estimated_savings_per_prompt:,} tokens")

        if result.blocks:
            st.divider()
            st.subheader("Detected Context Blocks")
            rows = [
                {
                    "Category":   b.category,
                    "Tokens":     b.token_count,
                    "Occurrences": b.occurrences,
                    "Content":    b.content[:100] + ("…" if len(b.content) > 100 else ""),
                }
                for b in result.blocks
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Category breakdown
            cat_counts = {}
            for b in result.blocks:
                cat_counts[b.category] = cat_counts.get(b.category, 0) + b.token_count
            fig = px.pie(
                pd.DataFrame({"Category": list(cat_counts.keys()),
                              "Tokens":   list(cat_counts.values())}),
                names="Category", values="Tokens",
                title="Token Distribution by Category",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4,
            )
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0", margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Generated CLAUDE.md")
        st.code(result.generated_content, language="markdown")
        st.download_button(
            "⬇ Download CLAUDE.md",
            data=result.generated_content,
            file_name="CLAUDE.md",
            mime="text/markdown",
        )


# ===========================================================================
# PAGE: RAG Advisor
# ===========================================================================

elif page == "RAG Advisor":
    st.title("🔍 RAG Advisor")
    st.caption("Detect large static context blocks and get retrieval-based recommendations.")

    prompt    = st.text_area("Prompt", height=260, placeholder="Paste your prompt here…")
    threshold = st.slider("Token threshold (flag blocks above this size)", 50, 1000, 300, step=50)
    run       = st.button("Analyse", type="primary")

    if run and prompt.strip():
        advice = analyze_rag_opportunities(prompt, threshold)
        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tokens",        f"{advice.total_prompt_tokens:,}")
        c2.metric("Context Tokens",      f"{advice.total_context_tokens:,}")
        c3.metric("Context Fraction",    f"{advice.context_fraction:.0f}%")
        c4.metric("Est. RAG Savings",    f"{advice.total_estimated_savings_pct:.0f}%")

        if advice.blocks:
            st.divider()
            st.subheader("Large Context Blocks")
            rows = [
                {
                    "Type":        b.block_type,
                    "Lines":       f"{b.start_line}–{b.end_line}",
                    "Tokens":      b.token_count,
                    "Est. Saving": f"{b.estimated_savings_pct:.0f}%",
                    "Suggestion":  b.retrieval_suggestion[:100] + "…",
                }
                for b in advice.blocks
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            type_counts = {}
            for b in advice.blocks:
                type_counts[b.block_type] = type_counts.get(b.block_type, 0) + b.token_count
            fig = px.bar(
                pd.DataFrame({"Type": list(type_counts.keys()),
                              "Tokens": list(type_counts.values())}),
                x="Type", y="Tokens", title="Context Tokens by Block Type",
                color_discrete_sequence=["#f59e0b"],
            )
            fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                              xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#334155"))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Recommendations")
        for r in advice.recommendations:
            st.info(r)

    elif run:
        st.warning("Please enter a prompt.")


# ===========================================================================
# PAGE: Example Pruner
# ===========================================================================

elif page == "Example Pruner":
    st.title("✂️ Example Pruner")
    st.caption("Cluster similar examples using TF-IDF cosine similarity and keep only representatives.")

    prompt    = st.text_area("Prompt", height=260, placeholder="Paste a prompt containing examples…")
    model     = st.selectbox("Model", MODELS, key="ep_model")
    threshold = st.slider("Similarity threshold", 0.1, 1.0, 0.65, step=0.05,
                          help="Examples above this similarity score are considered redundant.")
    run       = st.button("Prune", type="primary")

    if run and prompt.strip():
        result = prune_examples(prompt, threshold, model)
        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Examples Found",   len(result.original_examples))
        c2.metric("Clusters Formed",  len(result.clusters))
        c3.metric("Examples Kept",    len(result.kept_examples))
        c4.metric("Tokens Saved",     f"{result.tokens_saved:,}  ({result.reduction_pct:.1f}%)")

        if result.clusters:
            st.divider()
            st.subheader("Cluster Detail")
            rows = [
                {
                    "Representative":  c.representative[:80] + ("…" if len(c.representative) > 80 else ""),
                    "Members":         len(c.members),
                    "Avg Similarity":  f"{c.similarity_score:.2f}",
                    "Tokens Saved":    c.tokens_saved,
                }
                for c in result.clusters
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            fig = px.bar(
                pd.DataFrame({"Cluster": [f"#{i+1}" for i in range(len(result.clusters))],
                              "Tokens Saved": [c.tokens_saved for c in result.clusters]}),
                x="Cluster", y="Tokens Saved",
                title="Tokens Saved per Cluster",
                color_discrete_sequence=["#22c55e"],
            )
            fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                              xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#334155"))
            st.plotly_chart(fig, use_container_width=True)

        elif result.original_examples:
            st.success("No redundant examples detected at this threshold — all examples are sufficiently distinct.")
        else:
            st.warning("No examples detected in this prompt.")

    elif run:
        st.warning("Please enter a prompt.")


# ===========================================================================
# PAGE: Provider Comparison
# ===========================================================================

elif page == "Provider Comparison":
    st.title("💰 Provider Comparison")
    st.caption("Compare the cost of your prompt across Anthropic, OpenAI, Google, and Bedrock.")

    prompt = st.text_area("Prompt", height=180, placeholder="Paste your prompt here…")

    col_a, col_b = st.columns(2)
    with col_a:
        count_model = st.selectbox("Model for token counting", MODELS, key="pc_model")
    with col_b:
        all_providers = list(list_all_providers().keys())
        selected_providers = st.multiselect("Providers", all_providers,
                                            default=all_providers)
    run = st.button("Compare", type="primary")

    if run and prompt.strip():
        input_tok  = count_tokens(prompt, count_model)
        output_tok = estimate_output_tokens(prompt, count_model)
        comparison = compare_providers(input_tok, output_tok,
                                       selected_providers or None)
        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input Tokens",    f"{input_tok:,}")
        c2.metric("Output Tokens",   f"{output_tok:,}")
        c3.metric("Cheapest",        format_cost(comparison.cheapest.total_cost),
                  help=f"{comparison.cheapest.provider}/{comparison.cheapest.model}")
        c4.metric("Most Expensive",  format_cost(comparison.most_expensive.total_cost),
                  help=f"{comparison.most_expensive.provider}/{comparison.most_expensive.model}")

        st.divider()

        df = pd.DataFrame([
            {
                "Provider":      r.provider,
                "Model":         r.model,
                "Input Cost":    r.input_cost,
                "Output Cost":   r.output_cost,
                "Total Cost":    r.total_cost,
                "Context Window": f"{r.context_window:,}",
            }
            for r in comparison.rows
        ])

        # Chart
        fig = px.bar(
            df, x="Model", y="Total Cost", color="Provider",
            title="Total Cost by Model (sorted cheapest first)",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text_auto=".6f",
        )
        fig.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", xaxis_tickangle=-35,
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#334155"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table with formatted costs
        df["Input Cost"]  = df["Input Cost"].apply(format_cost)
        df["Output Cost"] = df["Output Cost"].apply(format_cost)
        df["Total Cost"]  = df["Total Cost"].apply(format_cost)
        st.dataframe(df, use_container_width=True, hide_index=True)

    elif run:
        st.warning("Please enter a prompt.")


# ===========================================================================
# PAGE: Budget Tracker
# ===========================================================================

elif page == "Budget Tracker":
    st.title("💳 Budget Tracker")
    st.caption("Set cost limits and monitor daily spend.")

    tab_cfg, tab_check, tab_history = st.tabs(["Configure", "Check Spend", "History"])

    # ── Configure ──────────────────────────────────────────────────────────
    with tab_cfg:
        st.subheader("Set Budget Limits")
        c1, c2, c3 = st.columns(3)
        with c1:
            session_budget = st.number_input("Session budget ($)", min_value=0.0,
                                             value=5.0, step=0.5, format="%.2f")
        with c2:
            daily_budget = st.number_input("Daily budget ($)", min_value=0.0,
                                           value=20.0, step=1.0, format="%.2f")
        with c3:
            warn_at = st.slider("Warn at (%)", 50, 95, 80)

        if st.button("Save Budget", type="primary"):
            from claude_toolkit.budget.budget import save_budget_config
            from claude_toolkit.models import BudgetConfig
            cfg = BudgetConfig(
                session_budget=session_budget if session_budget > 0 else None,
                daily_budget=daily_budget if daily_budget > 0 else None,
                warn_at_pct=float(warn_at),
            )
            save_budget_config(cfg)
            st.success("Budget saved to ~/.claude-toolkit/budget.json")

    # ── Check Spend ────────────────────────────────────────────────────────
    with tab_check:
        st.subheader("Evaluate a Session Cost")
        cost_input = st.number_input("Session cost ($)", min_value=0.0,
                                     value=0.0, step=0.01, format="%.4f")
        record = st.checkbox("Record in daily ledger")

        if st.button("Check Budget", type="primary"):
            cfg   = load_budget_config()
            alert = check_budget(cost_input, cfg, add_to_daily=record)

            color_map = {"ok": "success", "warn": "warning",
                         "critical": "error", "exceeded": "error"}
            getattr(st, color_map.get(alert.level, "info"))(alert.message)

            cols = st.columns(2)
            if alert.session_budget:
                cols[0].plotly_chart(
                    _gauge(alert.session_cost, alert.session_budget,
                           "Session Spend", " $"),
                    use_container_width=True,
                )
            if alert.daily_budget:
                cols[1].plotly_chart(
                    _gauge(alert.daily_cost, alert.daily_budget,
                           "Daily Spend", " $"),
                    use_container_width=True,
                )

    # ── History ────────────────────────────────────────────────────────────
    with tab_history:
        st.subheader("7-Day Spend History")
        days    = st.slider("Days to show", 7, 30, 7)
        history = get_spend_history(days)

        df_h = pd.DataFrame(
            {"Date": list(history.keys()), "Spend ($)": list(history.values())}
        )
        fig = px.bar(df_h, x="Date", y="Spend ($)",
                     title="Daily Spend",
                     color_discrete_sequence=["#6366f1"])
        fig.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", xaxis_tickangle=-30,
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#334155"),
        )
        st.plotly_chart(fig, use_container_width=True)

        total = sum(history.values())
        st.metric(f"Total spend ({days} days)", format_cost(total))
        st.dataframe(df_h, use_container_width=True, hide_index=True)
