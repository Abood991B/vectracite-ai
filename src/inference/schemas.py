from typing import Any, Dict, List
from pydantic import BaseModel, Field

from src.config import settings


class PageInput(BaseModel):
    page_number: int = Field(
        ..., ge=1, le=10_000, description="1-indexed page number in the document"
    )
    text: str = Field(
        ...,
        min_length=10,
        max_length=200_000,
        description="Page text and/or Markdown tables",
    )


class IngestRequest(BaseModel):
    doc_name: str = Field(
        ..., min_length=1, max_length=200, description="Document identifier (e.g., 'AMD_2022_10K')"
    )
    pages: List[PageInput] = Field(..., min_length=1, max_length=settings.MAX_STRUCTURED_PAGES)


class PdfUploadRequest(BaseModel):
    doc_name: str = Field(
        ..., min_length=1, max_length=200, description="Document name or filename"
    )
    pdf_base64: str = Field(
        ..., min_length=20, max_length=50_000_000, description="Base64-encoded PDF file content"
    )


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=4000, description="Analytical question across indexed documents"
    )
    top_k: int = Field(default=4, ge=1, le=20, description="Number of reranked chunks to retrieve")
    use_cache: bool = Field(default=True, description="Whether to check Redis semantic cache")
    doc_name: str | None = Field(
        default=None,
        max_length=200,
        description="Optional document name to scope search to a single PDF",
    )


class LLMConfigRequest(BaseModel):
    base_url: str | None = Field(default=None, min_length=1, max_length=2000)
    api_key: str | None = Field(default=None, max_length=1000)
    model_name: str | None = Field(default=None, min_length=1, max_length=200)
    enabled: bool | None = Field(
        default=None, description="Explicitly enable or disable the remote provider"
    )


class TableCellSchema(BaseModel):
    doc_name: str
    page_number: int
    row_key: str
    col_header: str
    value: str
    row_index: int = 0
    col_index: int = 0


class RowDeltaSchema(BaseModel):
    row_key: str
    comparison: str
    absolute_change: str
    pct_change: str


class CitationItem(BaseModel):
    chunk_id: str
    doc_name: str
    page_number: int
    contains_table: bool
    rerank_score: float
    excerpt: str


class NumericVerificationSchema(BaseModel):
    """Numeric token presence only; this does not verify claim meaning or accuracy."""
    is_fully_grounded: bool
    numeric_grounding_score: float
    verified_figures: List[str] = Field(default_factory=list)
    unverified_figures: List[str] = Field(default_factory=list)
    verification_applicable: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    matched_cells: List[TableCellSchema] = Field(default_factory=list)
    computed_deltas: List[RowDeltaSchema] = Field(default_factory=list)
    citations: List[CitationItem]
    verification: NumericVerificationSchema
    cache_hit: bool = False
    cache_similarity: float | None = None
    latency_ms: float
    guardrail_flags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
