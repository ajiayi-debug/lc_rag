import asyncio
import re
import uuid
from dataclasses import dataclass

import tiktoken
from anthropic import AsyncAnthropic

from config import (
    ANTHROPIC_API_KEY,
    CHILD_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    SUMMARIZE_MODEL,
    TICKER_TO_COMPANY,
)
from ingestion.pdf_parser import ParsedDocument

_enc = tiktoken.get_encoding("cl100k_base")
_async_anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# Max concurrent Haiku calls per document — stays within free-tier rate limits
# (moved Semaphore into _summarize_batch)


@dataclass
class Chunk:
    id: str
    text: str           # child text — what gets embedded & searched
    parent_text: str    # wider context — what gets sent to the LLM
    ticker: str
    company_name: str
    filing_type: str
    filing_date: str
    filename: str
    page_num: int
    section: str
    chunk_type: str     # "text" | "table" | "table_summary"
    chunk_index: int

    def to_payload(self) -> dict:
        return {
            "text": self.text,
            "parent_text": self.parent_text,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "filing_type": self.filing_type,
            "filing_date": self.filing_date,
            "filename": self.filename,
            "page_num": self.page_num,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
        }


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = _enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += max_tokens - overlap
    return chunks


def _extract_section(markdown: str) -> str:
    headers = re.findall(r"^#{1,4}\s+(.+)$", markdown, re.MULTILINE)
    return headers[-1].strip() if headers else "General"


# ── Async table summarization ─────────────────────────────────────────────────

async def _summarize_one(table_md: str, context: str, ticker: str, filing_type: str, sem: asyncio.Semaphore = None) -> str:
    prompt = (
        f"You are analyzing a table from a {filing_type} SEC filing by {ticker}.\n\n"
        f"Surrounding context:\n{context[:400]}\n\n"
        f"Table:\n{table_md}\n\n"
        "Write a 2-4 sentence prose summary of the key financial figures in this table. "
        "Include specific numbers and year-over-year comparisons where visible. "
        "Output only the summary, no preamble."
    )
    
    if sem:
        async with sem:
            response = await _async_anthropic.messages.create(
                model=SUMMARIZE_MODEL,
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
    else:
        response = await _async_anthropic.messages.create(
            model=SUMMARIZE_MODEL,
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
    return response.content[0].text.strip()


class _CreditError(Exception):
    """Raised when Anthropic returns a credit balance error — should abort ingestion immediately."""


async def _summarize_batch(
    table_data: list[dict],
) -> list[str | BaseException]:
    """Run all table summarizations for a document concurrently."""
    sem = asyncio.Semaphore(5)
    tasks = [
        _summarize_one(td["table_md"], td["context"], td["ticker"], td["filing_type"], sem)
        for td in table_data
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # If any result is a credit error, raise immediately so ingestion aborts
    for r in results:
        if isinstance(r, BaseException) and "credit balance" in str(r).lower():
            raise _CreditError(str(r))
    return results


# ── Main chunking ─────────────────────────────────────────────────────────────

def chunk_document(doc: ParsedDocument, summarize_tables: bool = True) -> list[Chunk]:
    company_name = TICKER_TO_COMPANY.get(doc.ticker, doc.ticker)
    chunks: list[Chunk] = []
    chunk_index = 0

    # ── Collect all tables across the document first ──────────────────────────
    table_jobs: list[dict] = []   # metadata for each table needing a summary
    table_raw_chunks: list[Chunk] = []  # raw table chunks (always added)

    for page in doc.pages:
        section = _extract_section(page.markdown)
        header = f"[{doc.ticker} | {doc.filing_type} | {doc.filing_date} | {section} | Page {page.page_num}]"

        for table_md in page.tables:
            table_text = f"{header}\n{table_md}"
            raw_chunk = Chunk(
                id=str(uuid.uuid4()),
                text=table_text,
                parent_text=table_text,
                ticker=doc.ticker,
                company_name=company_name,
                filing_type=doc.filing_type,
                filing_date=doc.filing_date,
                filename=doc.filename,
                page_num=page.page_num,
                section=section,
                chunk_type="table",
                chunk_index=0,  # assigned later
            )
            table_raw_chunks.append(raw_chunk)

            if summarize_tables and _count_tokens(table_md) > 40:
                table_jobs.append({
                    "table_md": table_md,
                    "context": page.text_without_tables,
                    "ticker": doc.ticker,
                    "filing_type": doc.filing_type,
                    "header": header,
                    "table_text": table_text,
                    "page_num": page.page_num,
                    "section": section,
                    "raw_chunk_ref": raw_chunk,
                })

    # ── Run all table summarizations in parallel ──────────────────────────────
    summaries: list[str | BaseException] = []
    if table_jobs:
        summaries = asyncio.run(_summarize_batch(table_jobs))

    # ── Add raw table chunks + summary chunks ─────────────────────────────────
    for raw_chunk in table_raw_chunks:
        raw_chunk.chunk_index = chunk_index
        chunks.append(raw_chunk)
        chunk_index += 1

    job_idx = 0
    for job in table_jobs:
        result = summaries[job_idx]
        job_idx += 1
        if isinstance(result, BaseException):
            print(f"  [warn] table summary failed {doc.filename} p{job['page_num']}: {result}")
            continue
        summary_text = f"{job['header']}\n{result}"
        chunks.append(Chunk(
            id=str(uuid.uuid4()),
            text=summary_text,
            parent_text=job["table_text"],   # parent = raw table
            ticker=doc.ticker,
            company_name=company_name,
            filing_type=doc.filing_type,
            filing_date=doc.filing_date,
            filename=doc.filename,
            page_num=job["page_num"],
            section=job["section"],
            chunk_type="table_summary",
            chunk_index=chunk_index,
        ))
        chunk_index += 1

    # ── Text chunks ───────────────────────────────────────────────────────────
    for page in doc.pages:
        section = _extract_section(page.markdown)
        header = f"[{doc.ticker} | {doc.filing_type} | {doc.filing_date} | {section} | Page {page.page_num}]"
        body = page.text_without_tables.strip()
        if not body:
            continue

        child_texts = _split_text(body, CHILD_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)

        for i, child in enumerate(child_texts):
            parent_parts: list[str] = []
            if i > 0:
                prev_tokens = _enc.encode(child_texts[i - 1])
                parent_parts.append(_enc.decode(prev_tokens[-150:]))
            parent_parts.append(child)
            if i < len(child_texts) - 1:
                next_tokens = _enc.encode(child_texts[i + 1])
                parent_parts.append(_enc.decode(next_tokens[:150]))

            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=f"{header}\n{child}",
                parent_text=f"{header}\n{' '.join(parent_parts)}",
                ticker=doc.ticker,
                company_name=company_name,
                filing_type=doc.filing_type,
                filing_date=doc.filing_date,
                filename=doc.filename,
                page_num=page.page_num,
                section=section,
                chunk_type="text",
                chunk_index=chunk_index,
            ))
            chunk_index += 1

    return chunks
