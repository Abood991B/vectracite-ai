from pathlib import Path
from pydantic import SecretStr
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
    CORPUS_INDEX_PATH: Path = DATA_DIR / "corpus_index.json"
    EMBEDDING_CACHE_PATH: Path = DATA_DIR / "corpus_embeddings.npz"
    USER_UPLOADS_DIR: Path = DATA_DIR / "uploads"

    # Parsing/chunking configuration. The corpus cache is keyed by these values.
    PARSER_CHUNK_SIZE: int = 1200
    PARSER_CHUNK_OVERLAP: int = 200
    EMBEDDING_CACHE_VERSION: int = 1

    # Qdrant Vector Store Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "financebench_hybrid_v1"
    QDRANT_IN_MEMORY: bool = False
    INDEX_SYNC_SECONDS: float = 15.0
    ALLOW_QDRANT_FALLBACK: bool = False
    HYBRID_DENSE_WEIGHT: float = 0.95
    HYBRID_BM25_WEIGHT: float = 0.05
    QDRANT_UPSERT_BATCH_SIZE: int = 256

    # Embedding & Cross-Encoder Reranker Configuration
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    USE_LIGHTWEIGHT_MODELS: bool = False

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
    LLM_REMOTE_ENABLED: bool = False
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_TOKENS: int = 800

    # API security. Local development remains open unless AUTH_REQUIRED is enabled.
    # Deployments should provide distinct read/admin keys through a secret manager.
    AUTH_REQUIRED: bool = False
    API_READ_KEY: SecretStr = SecretStr("")
    API_ADMIN_KEY: SecretStr = SecretStr("")
    CORS_ALLOWED_ORIGINS: str = "http://localhost:8080,http://127.0.0.1:8080"

    # Ingestion resource limits
    MAX_PDF_BYTES: int = 25 * 1024 * 1024
    MAX_PDF_PAGES: int = 500
    MAX_STRUCTURED_PAGES: int = 500
    MAX_STRUCTURED_TEXT_CHARS: int = 2_000_000
    INGEST_TIMEOUT_SECONDS: float = 120.0
    MAX_REQUEST_BYTES: int = 0
    MAX_RENDERED_PDF_PAGES: int = 5000
    REQUIRE_REDIS: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def effective_max_request_bytes(self) -> int:
        """Bound request bodies before JSON/base64 decoding allocates large buffers."""
        if self.MAX_REQUEST_BYTES > 0:
            return self.MAX_REQUEST_BYTES
        encoded_pdf = ((self.MAX_PDF_BYTES + 2) // 3) * 4
        structured_text = self.MAX_STRUCTURED_TEXT_CHARS * 4
        return max(encoded_pdf, structured_text) + 1_000_000

    def validate_auth_configuration(self) -> None:
        """Fail closed when authentication is enabled without distinct credentials."""
        if not self.AUTH_REQUIRED:
            return
        read_key = self.API_READ_KEY.get_secret_value()
        admin_key = self.API_ADMIN_KEY.get_secret_value()
        if not read_key or not admin_key:
            raise RuntimeError(
                "AUTH_REQUIRED=true requires both API_READ_KEY and API_ADMIN_KEY."
            )
        if read_key == admin_key:
            raise RuntimeError("API_READ_KEY and API_ADMIN_KEY must be distinct.")


settings = Settings()
settings.PDF_DIR.mkdir(parents=True, exist_ok=True)
settings.CORPUS_PDF_DIR.mkdir(parents=True, exist_ok=True)
settings.USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
