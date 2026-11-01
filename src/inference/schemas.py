from typing import Any, Dict, List
from pydantic import BaseModel, Field


class PageInput(BaseModel):
    page_number: int = Field(..., ge=1, description="1-indexed page number in the document")
    text: str = Field(..., min_length=10, description="Page text and/or Markdown tables")


class IngestRequest(BaseModel):
    doc_name: str = Field(..., description="Document identifier (e.g., 'AMD_2022_10K')")
    pages: List[PageInput] = Field(..., min_length=1)


class PdfUploadRequest(BaseModel):
    doc_name: str = Field(..., min_length=1, description="Document name or filename")
    pdf_base64: str = Field(..., min_length=20, description="Base64-encoded PDF file content")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Analytical question across indexed documents")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of reranked chunks to retrieve")
    use_cache: bool = Field(default=True, description="Whether to check Redis semantic cache")


class TableCellSchema(BaseModel):
    doc_name: str
    page_number: int
    row_key: str
    col_header: str
    value: str


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
    is_fully_grounded: bool
    numeric_grounding_score: float
    verified_figures: List[str] = Field(default_factory=list)
    unverified_figures: List[str] = Field(default_factory=list)


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
