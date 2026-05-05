import re
from pathlib import Path
from dataclasses import dataclass, field
import pymupdf
import pymupdf4llm

# Suppress MuPDF's stderr noise about tiny/unscalable images in PDFs
pymupdf.TOOLS.mupdf_display_errors(False)


@dataclass
class PageContent:
    page_num: int           # 1-indexed
    markdown: str           # full page markdown from pymupdf4llm
    tables: list[str]       # list of raw markdown table strings detected on page
    text_without_tables: str


@dataclass
class ParsedDocument:
    ticker: str
    filing_type: str        # 10-K, 10-Q, 8-K
    filing_date: str        # YYYY-MM-DD
    filename: str
    filepath: str
    pages: list[PageContent]


def extract_tables_from_markdown(markdown: str) -> tuple[list[str], str]:
    """Split markdown into tables (atomic) and non-table text."""
    lines = markdown.split("\n")
    tables: list[str] = []
    remaining: list[str] = []
    i = 0

    while i < len(lines):
        if re.match(r"\s*\|", lines[i]):
            table_lines: list[str] = []
            while i < len(lines) and re.match(r"\s*\|", lines[i]):
                table_lines.append(lines[i])
                i += 1
            # Only keep as table if it has header + separator (at least 2 rows)
            if len(table_lines) >= 2:
                tables.append("\n".join(table_lines))
            else:
                remaining.extend(table_lines)
        else:
            remaining.append(lines[i])
            i += 1

    return tables, "\n".join(remaining)


def parse_filename(filename: str) -> tuple[str, str, str]:
    """Parse TICKER_FILING-TYPE_DATE.pdf → (ticker, filing_type, filing_date)."""
    stem = filename.replace(".pdf", "")
    parts = stem.split("_")
    ticker = parts[0]
    filing_type = parts[1] if len(parts) > 1 else "UNKNOWN"
    raw_date = parts[2] if len(parts) > 2 else ""
    # Convert YYYYMMDD → YYYY-MM-DD
    if len(raw_date) == 8 and raw_date.isdigit():
        filing_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
    else:
        filing_date = raw_date
    return ticker, filing_type, filing_date


def parse_pdf(filepath: str | Path) -> ParsedDocument:
    filepath = Path(filepath)
    ticker, filing_type, filing_date = parse_filename(filepath.name)

    # pymupdf4llm gives per-page markdown with table detection built-in
    # Disable OCR to prevent stalling on tiny unscalable logos and graphs in SEC filings
    page_chunks = pymupdf4llm.to_markdown(str(filepath), page_chunks=True, use_ocr=False)

    pages: list[PageContent] = []
    for chunk in page_chunks:
        meta = chunk.get("metadata", {})
        # pymupdf4llm uses 0-indexed pages
        page_num = meta.get("page", 0) + 1
        markdown = chunk.get("text", "")

        tables, text_without_tables = extract_tables_from_markdown(markdown)

        pages.append(PageContent(
            page_num=page_num,
            markdown=markdown,
            tables=tables,
            text_without_tables=text_without_tables,
        ))

    return ParsedDocument(
        ticker=ticker,
        filing_type=filing_type,
        filing_date=filing_date,
        filename=filepath.name,
        filepath=str(filepath),
        pages=pages,
    )
