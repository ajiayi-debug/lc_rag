import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("claude")
VOYAGE_API_KEY = os.getenv("voyage")
COHERE_API_KEY = os.getenv("cohere")

ROOT_DIR = Path(__file__).parent
SEC_FILINGS_DIR = ROOT_DIR / "sec_filings_pdf"
QDRANT_PATH = str(ROOT_DIR / "qdrant_storage")
BM25_INDEX_PATH = str(ROOT_DIR / "bm25_index.pkl")

# Voyage AI — finance-specific model, tuned for SEC filings
EMBEDDING_MODEL = "voyage-finance-2"
EMBEDDING_DIM = 1024

# Local reranking — free, no API, ~110MB download on first run
# rank-T5-flan outperforms ms-marco-MiniLM on nDCG while staying within flashrank
RERANK_MODEL = "rank-T5-flan"

# Claude models
LLM_MODEL = "claude-haiku-4-5-20251001"           # used for generation + query analysis
SUMMARIZE_MODEL = "claude-haiku-4-5-20251001"  # used for table summarization 

CHILD_CHUNK_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50

TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "30"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "6"))
MAX_CHUNKS_TOTAL = int(os.getenv("MAX_CHUNKS_TOTAL", "12"))
MAX_SELF_CORRECT_RETRIES = 2

# Feature flags (set via env vars for per-experiment overrides)
DISABLE_COHERE = os.getenv("DISABLE_COHERE", "0") == "1"
DISABLE_HYDE = os.getenv("DISABLE_HYDE", "0") == "1"

COLLECTION_NAME = "sec_filings"

TICKER_TO_COMPANY = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOG": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "AVGO": "Broadcom Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "UNH": "UnitedHealth Group Inc.",
    "LLY": "Eli Lilly and Company",
    "V": "Visa Inc.",
    "XOM": "Exxon Mobil Corporation",
    "MA": "Mastercard Inc.",
    "JNJ": "Johnson & Johnson",
    "COST": "Costco Wholesale Corporation",
    "WMT": "Walmart Inc.",
    "PG": "Procter & Gamble Co.",
    "HD": "The Home Depot Inc.",
    "BAC": "Bank of America Corporation",
    "ORCL": "Oracle Corporation",
    "MRK": "Merck & Co. Inc.",
    "ABBV": "AbbVie Inc.",
    "CVX": "Chevron Corporation",
    "KO": "The Coca-Cola Company",
    "NFLX": "Netflix Inc.",
    "AMD": "Advanced Micro Devices Inc.",
    "PEP": "PepsiCo Inc.",
    "CSCO": "Cisco Systems Inc.",
    "WFC": "Wells Fargo & Company",
    "MS": "Morgan Stanley",
    "GS": "Goldman Sachs Group Inc.",
    "IBM": "IBM Corporation",
    "MCD": "McDonald's Corporation",
    "GE": "GE Aerospace",
    "GEV": "GE Vernova Inc.",
    "AXP": "American Express Company",
    "RTX": "RTX Corporation",
    "PLTR": "Palantir Technologies Inc.",
    "CAT": "Caterpillar Inc.",
    "TXN": "Texas Instruments Inc.",
    "INTC": "Intel Corporation",
    "MU": "Micron Technology Inc.",
    "AMAT": "Applied Materials Inc.",
    "LRCX": "Lam Research Corporation",
    "T": "AT&T Inc.",
    "TMUS": "T-Mobile US Inc.",
    "VZ": "Verizon Communications Inc.",
    "PM": "Philip Morris International Inc.",
    "BRK-B": "Berkshire Hathaway Inc.",
}
