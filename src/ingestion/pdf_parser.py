"""
PDF parsing using Docling.
Extracts text, tables, page numbers, and section headers while preserving structure.
Uses document.iterate_items() which yields (item, level) tuples.
"""

from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from src.utils.logger import get_logger

logger = get_logger(__name__)

# DoclingDocument.iterate_items() yields (item, level). Item can be SectionHeaderItem, TextItem, TableItem, etc.
# We use getattr for compatibility across docling versions.


def _page_no(item: Any) -> int:
    prov = getattr(item, "prov", None) or []
    if prov and len(prov) > 0:
        p = prov[0]
        return getattr(p, "page_no", 0) if hasattr(p, "page_no") else (p.get("page_no", 0) if isinstance(p, dict) else 0)
    return 0


def _item_label(item: Any) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return ""
    return str(getattr(label, "value", label))


def _item_text(item: Any) -> str:
    return getattr(item, "text", "") or ""


def parse_pdf_with_docling(
    pdf_path: Path,
    extract_tables: bool = True,
) -> list[dict[str, Any]]:
    """
    Parse a PDF with Docling and return a list of structured sections.
    Each section includes: text, tables, page number, section headers.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = extract_tables

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(str(pdf_path))
    document = result.document

    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None
    current_text: list[str] = []
    current_tables: list[str] = []

    def flush_section():
        nonlocal current_section, current_text, current_tables
        if current_section is not None and (current_text or current_tables):
            current_section["content"] = "\n\n".join(current_text).strip()
            current_section["table_content"] = "\n\n".join(current_tables).strip()
            sections.append(current_section)
        current_section = None
        current_text = []
        current_tables = []

    # DoclingDocument.iterate_items() yields (item, level) — item is a NodeItem/DocItem, not a tuple
    try:
        iterator = document.iterate_items(with_groups=False)
    except AttributeError:
        iterator = []

    for entry in iterator:
        # Handle (item, level) tuple from iterate_items()
        if isinstance(entry, tuple) and len(entry) >= 1:
            item = entry[0]
        else:
            item = entry
        if not hasattr(item, "prov"):
            continue
        item_type = _item_label(item)
        page_no = _page_no(item)
        is_heading = (
            "section" in item_type.lower()
            or "heading" in item_type.lower()
            or item_type == "subtitle-level-1"
            or "title" in item_type.lower()
        )
        is_text = (
            "text" in item_type.lower()
            or "paragraph" in item_type.lower()
            or "list" in item_type.lower()
            or "caption" in item_type.lower()
            or "footnote" in item_type.lower()
        )
        is_table = "table" in item_type.lower()

        if is_heading:
            flush_section()
            title = _item_text(item)
            current_section = {
                "section_title": title.strip() or "Section",
                "page_number": page_no,
                "content": "",
                "table_content": "",
            }
        elif is_text:
            text = _item_text(item)
            if getattr(item, "marker", None):
                text = f"{item.marker} {text}".strip()
            if text.strip():
                current_text.append(text.strip())
            if current_section is None:
                current_section = {
                    "section_title": "Introduction",
                    "page_number": page_no or 1,
                    "content": "",
                    "table_content": "",
                }
        elif is_table and extract_tables:
            table_text = ""
            if hasattr(item, "export_to_markdown"):
                try:
                    table_text = item.export_to_markdown(doc=document)
                except Exception:
                    table_text = str(getattr(item, "text", ""))
            if table_text.strip():
                current_tables.append(table_text.strip())
            if current_section is None:
                current_section = {
                    "section_title": "Tables",
                    "page_number": page_no or 1,
                    "content": "",
                    "table_content": "",
                }

    flush_section()

    # Fallback: if no sections from structure, use full-doc markdown
    if not sections:
        try:
            full_md = document.export_to_markdown()
            if full_md and full_md.strip():
                sections = [
                    {
                        "section_title": "Full Document",
                        "page_number": 1,
                        "content": full_md,
                        "table_content": "",
                    }
                ]
        except Exception:
            pass

    logger.info("Parsed %s: %d sections", pdf_path.name, len(sections))
    return sections
