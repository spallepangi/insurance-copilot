"""
Run the RAG pipeline interactively (CLI) or start API/UI.
Usage:
  python -m scripts.run_pipeline              # interactive CLI
  python -m scripts.run_pipeline api          # start FastAPI
  python -m scripts.run_pipeline ui          # start Streamlit
  python -m scripts.run_pipeline latency-stats # print latency stats from logs
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_cli():
    from src.rag.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    print("InsuranceCopilot AI — type a question or 'compare <question>' or 'quit'.")
    while True:
        try:
            line = input("You: ").strip()
            if not line or line.lower() in ("quit", "exit", "q"):
                break
            if line.lower().startswith("compare "):
                q = line[8:].strip()
                out = pipeline.compare_plans(question=q)
            else:
                out = pipeline.query(question=line)
            print("Answer:", out.get("answer", ""))
            for c in out.get("citations", [])[:3]:
                print(f"  — {c.get('plan')} | {c.get('section')} | p.{c.get('page')}")
            print(f"  [latency: {out.get('latency_ms')} ms, cost: ${out.get('cost', 0):4f}]")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)


def run_api():
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)


def run_ui():
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "src" / "ui" / "streamlit_app.py"),
        "--server.headless", "true",
    ], cwd=str(ROOT))


def run_latency_stats():
    from src.monitoring.latency_tracker import LatencyTracker
    stats = LatencyTracker.get_stats()
    print("Latency statistics (ms):")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cli"
    if cmd == "api":
        run_api()
    elif cmd == "ui":
        run_ui()
    elif cmd == "latency-stats":
        run_latency_stats()
    else:
        run_cli()


if __name__ == "__main__":
    main()
