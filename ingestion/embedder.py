import time
import tiktoken
import voyageai
from config import EMBEDDING_MODEL, VOYAGE_API_KEY

_client = voyageai.Client(api_key=VOYAGE_API_KEY)
_tokenizer = tiktoken.get_encoding("cl100k_base")

_MAX_RETRIES = 6
_RETRY_BASE = 20  # seconds — backs off exponentially on rate limit
_MAX_TOKENS_PER_BATCH = 100_000  # Voyage limit is 120k; keep headroom
_MAX_TEXTS_PER_BATCH = 128       # Voyage hard limit on number of texts


def _token_count(text: str) -> int:
    return len(_tokenizer.encode(text, disallowed_special=()))


def _embed_with_retry(texts: list[str], input_type: str) -> list[list[float]]:
    # If batch is too large to send as one, split and recurse
    if len(texts) > 1:
        for attempt in range(_MAX_RETRIES):
            try:
                return _client.embed(texts, model=EMBEDDING_MODEL, input_type=input_type).embeddings
            except voyageai.error.RateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _RETRY_BASE * (2 ** attempt)
                print(f"  [rate limit] waiting {wait}s before retry ({attempt + 1}/{_MAX_RETRIES})...")
                time.sleep(wait)
            except Exception as e:
                if "max allowed tokens" in str(e).lower() or "token" in str(e).lower():
                    mid = len(texts) // 2
                    print(f"  [token limit] batch too large ({len(texts)} texts) — splitting in half and retrying...")
                    left = _embed_with_retry(texts[:mid], input_type)
                    right = _embed_with_retry(texts[mid:], input_type)
                    return left + right
                raise
    else:
        return _client.embed(texts, model=EMBEDDING_MODEL, input_type=input_type).embeddings
    return []  # unreachable


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    batch: list[str] = []
    batch_tokens = 0

    for text in texts:
        t = _token_count(text)
        if batch and (batch_tokens + t > _MAX_TOKENS_PER_BATCH or len(batch) >= _MAX_TEXTS_PER_BATCH):
            all_embeddings.extend(_embed_with_retry(batch, input_type))
            batch, batch_tokens = [], 0
        batch.append(text)
        batch_tokens += t

    if batch:
        all_embeddings.extend(_embed_with_retry(batch, input_type))

    return all_embeddings


def embed_query(query: str) -> list[float]:
    return _embed_with_retry([query], "query")[0]
