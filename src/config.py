from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = ROOT / "data" / "data.pdf"
FINANCIAL_REPORT_PATH = ROOT / "data" / "financial_report.pdf"
INDEX_PATH = ROOT / "faiss_index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"   

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
BREAKPOINT_THRESHOLD_TYPE = "standard_deviation"