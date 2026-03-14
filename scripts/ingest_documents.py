"""
Ingestion script: parse PDFs, chunk, embed, insert into Qdrant.
Run from project root after setting .env (do not run automatically).
  python -m scripts.ingest_documents
"""

import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.ingestion.chunker import SectionChunker
from src.ingestion.metadata_extractor import extract_metadata_from_path
from src.ingestion.pdf_parser import parse_pdf_with_docling
from src.utils.config import DATA_DIR
from src.utils.logger import get_logger
from src.vector_store.index_builder import IndexBuilder

logger = get_logger(__name__)


def get_pdf_paths() -> list[Path]:
    """Collect PDFs from data/ (bronze.pdf, silver.pdf, gold.pdf, platinum.pdf or bbbronze etc.)."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}. Create it and add PDFs.")
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}. Add bronze.pdf, silver.pdf, gold.pdf, platinum.pdf.")
    return sorted(pdfs)


def main():
    logger.info("Starting ingestion from %s", DATA_DIR)
    pdf_paths = get_pdf_paths()
    chunker = SectionChunker()
    all_chunks = []
    for path in tqdm(pdf_paths, desc="Parsing PDFs"):
        try:
            sections = parse_pdf_with_docling(path)
            meta = extract_metadata_from_path(path)
            plan_name = meta["plan_name"]
            source_file = meta["source_file"]
            chunks = chunker.chunk_document(sections, plan_name=plan_name, source_file=source_file)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.exception("Failed to process %s: %s", path, e)
            raise
    if not all_chunks:
        raise RuntimeError("No chunks produced. Check PDF paths and parser.")
    logger.info("Total chunks: %d", len(all_chunks))
    builder = IndexBuilder()
    builder.index_chunks(all_chunks)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
