"""
Streamlit UI: search bar, plan comparison, answer display with citations, metrics dashboard.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src.*` imports resolve on Streamlit Cloud
# (Streamlit adds the file's directory to sys.path, not the repo root)
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st
from src.rag.rag_pipeline import RAGPipeline
from src.monitoring.latency_tracker import LatencyTracker

st.set_page_config(
    page_title="InsuranceCopilot AI",
    page_icon="📋",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/health-insurance.png", width=64)
    st.title("InsuranceCopilot AI")
    st.markdown("Healthcare insurance policy Q&A powered by RAG + BGE embeddings.")
    st.divider()
    st.markdown("**Plans available**")
    for plan, color in [("Bronze", "#cd7f32"), ("Silver", "#aaa9ad"), ("Gold", "#d4af37"), ("Platinum", "#e5e4e2")]:
        st.markdown(f"<span style='color:{color}'>⬤</span> {plan}", unsafe_allow_html=True)
    st.divider()
    st.caption("Model: gpt-4o-mini · Embeddings: BGE-large-en · Reranker: BGE-large")


@st.cache_resource(show_spinner="Loading RAG pipeline (models download on first run)...")
def get_pipeline():
    return RAGPipeline()


def _render_citations(citations: list[dict], max_excerpt: int = 350):
    if not citations:
        return
    st.subheader("Sources")
    for i, c in enumerate(citations, 1):
        plan = c.get("plan", "")
        section = c.get("section", "")
        page = c.get("page", "")
        excerpt = (c.get("policy_excerpt") or "")[:max_excerpt]
        with st.expander(f"[{i}] {plan} — {section} (p.{page})"):
            st.caption(excerpt + ("..." if len(c.get("policy_excerpt", "")) > max_excerpt else ""))


def _render_stage_timings(stage_timings: dict):
    if not stage_timings:
        return
    cols = st.columns(len(stage_timings))
    labels = {"retrieval": "Retrieval", "rerank": "Rerank", "compression": "Compress", "generation": "Generate", "total": "Total"}
    for col, (k, v) in zip(cols, stage_timings.items()):
        col.metric(labels.get(k, k), f"{v/1000:.2f}s")


def main():
    pipeline = get_pipeline()

    tab_query, tab_compare, tab_metrics = st.tabs(["Single Query", "Plan Comparison", "System Metrics"])

    # ── Single Query ─────────────────────────────────────────────────────────
    with tab_query:
        st.markdown("### Ask a policy question")
        col_q, col_p = st.columns([3, 1])
        question = col_q.text_input(
            "Your question",
            placeholder="e.g. Does the Bronze plan cover emergency room visits?",
            label_visibility="collapsed",
        )
        plan_filter = col_p.selectbox(
            "Plan filter",
            ["All plans", "Bronze", "Silver", "Gold", "Platinum"],
            label_visibility="collapsed",
        )
        plan_filter_val = None if plan_filter == "All plans" else plan_filter

        if st.button("Search", type="primary", use_container_width=False) and question.strip():
            with st.spinner("Retrieving and generating answer..."):
                try:
                    result = pipeline.query(question=question.strip(), plan_filter=plan_filter_val)
                    st.success("Answer")
                    st.markdown(result.get("answer", "No answer generated."))
                    st.divider()
                    _render_stage_timings(result.get("stage_timings_ms", {}))
                    st.caption(f"Total: {result.get('latency_ms', 0):.0f} ms · Cost: ${result.get('cost', 0):.5f}")
                    _render_citations(result.get("citations", []))
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Plan Comparison ───────────────────────────────────────────────────────
    with tab_compare:
        st.markdown("### Compare coverage across plans")
        question_c = st.text_input(
            "Comparison question",
            placeholder="e.g. Compare emergency room coverage between Bronze and Platinum.",
            label_visibility="collapsed",
            key="compare_q",
        )
        plans_selected = st.multiselect(
            "Plans to compare",
            ["Bronze", "Silver", "Gold", "Platinum"],
            default=["Bronze", "Silver", "Gold", "Platinum"],
        )
        if st.button("Compare Plans", type="primary") and question_c.strip():
            with st.spinner("Running parallel retrieval across plans..."):
                try:
                    result = pipeline.compare_plans(
                        question=question_c.strip(),
                        plans=plans_selected if plans_selected else None,
                    )
                    st.success("Comparison")
                    st.markdown(result.get("answer", ""))
                    st.divider()
                    _render_stage_timings(result.get("stage_timings_ms", {}))
                    st.caption(f"Total: {result.get('latency_ms', 0):.0f} ms · Cost: ${result.get('cost', 0):.5f}")

                    # Per-plan chunk breakdown
                    comp = result.get("comparison_data", {})
                    chunks_by_plan = comp.get("chunks_by_plan", {})
                    if chunks_by_plan:
                        st.divider()
                        st.markdown("**Retrieved chunks by plan**")
                        plan_cols = st.columns(len(chunks_by_plan))
                        for col, (plan, chunks) in zip(plan_cols, chunks_by_plan.items()):
                            col.markdown(f"**{plan}** ({len(chunks)} chunks)")
                            for c in chunks:
                                col.caption(f"Section: {c.get('section','?')} | p.{c.get('page','?')} | score: {c.get('score',0):.3f}")

                    _render_citations(result.get("citations", []))
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── System Metrics ────────────────────────────────────────────────────────
    with tab_metrics:
        st.markdown("### System Metrics")
        stats = LatencyTracker.get_stats()
        if stats.get("count", 0) == 0:
            st.info("No queries yet. Run a search to see latency stats.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Queries", stats["count"])
            m2.metric("p50 latency", f"{stats['p50_ms']:.0f} ms" if stats['p50_ms'] else "—")
            m3.metric("p95 latency", f"{stats['p95_ms']:.0f} ms" if stats['p95_ms'] else "—")
            m4.metric("p99 latency", f"{stats['p99_ms']:.0f} ms" if stats['p99_ms'] else "—")
            if stats.get("last_stage_timings_ms"):
                st.divider()
                st.markdown("**Last query stage breakdown**")
                _render_stage_timings(stats["last_stage_timings_ms"])
        if st.button("Refresh"):
            st.rerun()


if __name__ == "__main__":
    main()
