from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, type-safe application configuration loaded from environment variables."""

    PROJECT_NAME: str = "VectraCite-AI"
    VERSION: str = "1.0.0"

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PDF_DIR: Path = DATA_DIR / "pdfs"
    CORPUS_PDF_DIR: Path = DATA_DIR / "corpus_pdfs"

    # Qdrant Vector Store Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "financebench_hybrid_v1"
    QDRANT_IN_MEMORY: bool = True

    # Embedding & Cross-Encoder Reranker Configuration
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # Redis Semantic Cache Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CACHE_TTL_SECONDS: int = 86400

    # MLflow Experiment Tracking Configuration
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "VectraCite-FinanceBench-Evaluation"

    # Unified LLM Inference Endpoint (Self-Hosted vLLM or OpenAI-Compatible API)
    LLM_BASE_URL: str = "http://localhost:8000/v1"
    LLM_API_KEY: str = "EMPTY"
    LLM_MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
settings.PDF_DIR.mkdir(parents=True, exist_ok=True)
settings.CORPUS_PDF_DIR.mkdir(parents=True, exist_ok=True)
