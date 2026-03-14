# InsuranceCopilot AI — Workspace structure

Open this file or use **View → Explorer** (or `Cmd+Shift+E` / `Ctrl+Shift+E`) to see files in the sidebar.

```
insurance-copilot/
├── .env                    # Your API keys (do not commit)
├── .env.example
├── README.md
├── requirements.txt
├── WORKSPACE_TREE.md        # This file
│
├── data/                   # Put PDFs here: bronze.pdf, silver.pdf, gold.pdf, platinum.pdf
│   └── .gitkeep
│
├── logs/                   # Query metrics written here at runtime
│   └── .gitkeep
│
├── scripts/
│   ├── ingest_documents.py
│   ├── run_pipeline.py
│   └── compute_latency_stats.py
│
└── src/
    ├── __init__.py
    ├── api/
    │   ├── __init__.py
    │   └── app.py          # FastAPI backend
    ├── embeddings/
    │   ├── __init__.py
    │   └── embedder.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── evaluation_dataset.json
    │   ├── evaluation_runner.py
    │   └── metrics.py
    ├── ingestion/
    │   ├── __init__.py
    │   ├── chunker.py
    │   ├── metadata_extractor.py
    │   └── pdf_parser.py
    ├── monitoring/
    │   ├── __init__.py
    │   ├── cost_tracker.py
    │   ├── latency_tracker.py
    │   └── metrics_logger.py
    ├── rag/
    │   ├── __init__.py
    │   ├── answer_generator.py
    │   ├── plan_comparator.py
    │   └── rag_pipeline.py
    ├── retrieval/
    │   ├── __init__.py
    │   ├── hybrid_search.py
    │   ├── reranker.py
    │   └── retriever.py
    ├── ui/
    │   ├── __init__.py
    │   └── streamlit_app.py
    ├── utils/
    │   ├── __init__.py
    │   ├── config.py
    │   └── logger.py
    └── vector_store/
        ├── __init__.py
        ├── index_builder.py
        └── qdrant_client.py
```
