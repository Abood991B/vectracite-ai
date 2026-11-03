import os
import shutil
import tempfile
from pathlib import Path

os.environ["USE_LIGHTWEIGHT_MODELS"] = "true"
os.environ["QDRANT_IN_MEMORY"] = "true"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.config import settings

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
FIXTURE_CORPUS = FIXTURE_ROOT / "corpus_pdfs"
TEST_DATA = Path(tempfile.mkdtemp(prefix="vectracite-tests-"))
TEST_DATA.mkdir(parents=True, exist_ok=True)
shutil.copy(FIXTURE_ROOT / "benchmark_queries.json", TEST_DATA / "benchmark_queries.json")

settings.DATA_DIR = TEST_DATA
settings.PDF_DIR = TEST_DATA / "pdfs"
settings.CORPUS_PDF_DIR = FIXTURE_CORPUS
settings.CORPUS_INDEX_PATH = TEST_DATA / "corpus_index.json"
settings.EMBEDDING_CACHE_PATH = TEST_DATA / "corpus_embeddings.npz"
settings.USER_UPLOADS_DIR = TEST_DATA / "uploads"
settings.MLFLOW_TRACKING_URI = f"file:{(TEST_DATA / 'mlruns').as_posix()}"
settings.PDF_DIR.mkdir(parents=True, exist_ok=True)
settings.USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
