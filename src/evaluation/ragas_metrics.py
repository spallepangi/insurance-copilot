"""
Optional RAGAS (RAG Assessment) metrics: faithfulness, answer relevancy.
Requires: pip install ragas datasets (and OPENAI_API_KEY for RAGAS model calls).
When run with --rag --ragas, evaluation_runner will add ragas_faithfulness, ragas_answer_relevancy to metrics.
"""

from typing import List


def _contexts_from_result(result: dict) -> List[str]:
    """Build list of context strings from retrieved_chunks."""
    chunks = result.get("retrieved_chunks") or []
    return [(c.get("text") or "").strip() for c in chunks if (c.get("text") or "").strip()]


def compute_ragas_metrics(results: List[dict]) -> dict[str, float]:
    """
    Compute RAGAS metrics (faithfulness, answer_relevancy) when answers and contexts exist.
    Returns {} if ragas is not installed or no valid rows. Requires OPENAI_API_KEY (RAGAS uses LLM for scoring).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics._faithfulness import faithfulness
        from ragas.metrics._answer_relevance import answer_relevancy
    except ImportError:
        return {}

    rows = []
    for r in results:
        question = (r.get("query") or "").strip()
        answer = (r.get("answer") or "").strip()
        contexts = _contexts_from_result(r)
        if not question or not answer:
            continue
        rows.append({"question": question, "answer": answer, "contexts": contexts})

    if not rows:
        return {}

    try:
        # RAGAS expects user_input, response, retrieved_contexts (per SingleTurnSample)
        dataset = Dataset.from_list([
            {"user_input": x["question"], "response": x["answer"], "retrieved_contexts": x["contexts"]}
            for x in rows
        ])
        # Legacy metrics (faithfulness, answer_relevancy) accept llm/embeddings via evaluate()
        # answer_relevancy expects Langchain-style embeddings (embed_query); ragas wraps them
        import os
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {}
        client = OpenAI(api_key=api_key)
        from ragas.llms import llm_factory
        from langchain_openai import OpenAIEmbeddings
        llm = llm_factory("gpt-4o-mini", client=client)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )
        # EvaluationResult has _repr_dict: metric_name -> mean score
        if hasattr(result, "_repr_dict") and result._repr_dict:
            return {f"ragas_{k}": float(v) for k, v in result._repr_dict.items() if isinstance(v, (int, float))}
        if hasattr(result, "scores") and result.scores:
            keys = list(result.scores[0].keys()) if result.scores else []
            out = {}
            for k in keys:
                vals = [s[k] for s in result.scores if isinstance(s.get(k), (int, float))]
                if vals:
                    out[f"ragas_{k}"] = sum(vals) / len(vals)
            return out
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("RAGAS evaluation failed: %s", e)
    return {}
