"""
Streamlit UI: search bar, plan comparison, answer display with citations.
"""

import streamlit as st
from src.rag.rag_pipeline import RAGPipeline
from src.monitoring.latency_tracker import LatencyTracker


st.set_page_config(
    page_title="InsuranceCopilot AI",
    page_icon="📋",
    layout="centered",
)


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


def main():
    st.title("InsuranceCopilot AI")
    st.markdown("Ask questions about your healthcare insurance policy or compare plans.")

    pipeline = get_pipeline()
    mode = st.radio("Mode", ["Single query", "Plan comparison"], horizontal=True)

    if mode == "Single query":
        question = st.text_input("Your question", placeholder="e.g. Does the Bronze plan cover emergency room visits?")
        plan_filter = st.selectbox(
            "Filter by plan (optional)",
            ["All", "Bronze", "Silver", "Gold", "Platinum"],
        )
        plan_filter = None if plan_filter == "All" else plan_filter
        if st.button("Search") and question:
            with st.spinner("Searching..."):
                try:
                    result = pipeline.query(question=question, plan_filter=plan_filter)
                    st.subheader("Answer")
                    st.markdown(result.get("answer", "No answer generated."))
                    st.caption(f"Latency: {result.get('latency_ms', 0):.0f} ms | Cost: ${result.get('cost', 0):.4f}")
                    citations = result.get("citations", [])
                    if citations:
                        st.subheader("Citations")
                        for i, c in enumerate(citations, 1):
                            st.markdown(
                                f"**[{i}] {c.get('plan', '')}** — {c.get('section', '')} (p.{c.get('page', '')})"
                            )
                            st.text(c.get("policy_excerpt", "")[:400] + ("..." if len(c.get("policy_excerpt", "")) > 400 else ""))
                except Exception as e:
                    st.error(str(e))
    else:
        question = st.text_input(
            "Comparison question",
            placeholder="e.g. Compare emergency room coverage between Bronze and Platinum.",
        )
        if st.button("Compare") and question:
            with st.spinner("Comparing plans..."):
                try:
                    result = pipeline.compare_plans(question=question)
                    st.subheader("Comparison")
                    st.markdown(result.get("answer", ""))
                    st.caption(f"Latency: {result.get('latency_ms', 0):.0f} ms | Cost: ${result.get('cost', 0):.4f}")
                    citations = result.get("citations", [])
                    if citations:
                        st.subheader("Sources")
                        for i, c in enumerate(citations, 1):
                            st.markdown(f"**{c.get('plan', '')}** — {c.get('section', '')} (p.{c.get('page', '')})")
                            st.text(c.get("policy_excerpt", "")[:300] + "...")
                except Exception as e:
                    st.error(str(e))

    with st.expander("System metrics"):
        stats = LatencyTracker.get_stats()
        st.json(stats)


if __name__ == "__main__":
    main()
