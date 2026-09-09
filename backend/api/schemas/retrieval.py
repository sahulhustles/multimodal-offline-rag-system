"""Pydantic request/response models for the Phase 4 retrieval API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class SearchRequestSchema(BaseModel):
    """Request body for POST /api/v1/retrieval/search."""

    query: str = Field(..., min_length=1, max_length=2000, description="Natural-language search query")
    source_document_ids: list[str] = Field(
        ..., min_length=1, description="Source document IDs to search within"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Max results to return")
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum cosine similarity score"
    )
    modality_filter: Optional[str] = Field(
        default=None, description="Optional modality filter: 'text', 'image', or 'audio'"
    )


# ---------------------------------------------------------------------------
# Response — linked record
# ---------------------------------------------------------------------------


class LinkedRecordSchema(BaseModel):
    """A cross-modal linked record (e.g. CLIP ↔ LLaVA description pair)."""

    chunk_id: str
    source_type: str
    modality: str
    text_preview: Optional[str] = None
    llava_description: Optional[str] = None


# ---------------------------------------------------------------------------
# Response — single result
# ---------------------------------------------------------------------------


class RetrievalResultSchema(BaseModel):
    """A single hydrated retrieval result with full metadata and linked records."""

    chunk_id: str
    score: float
    source_document_id: str
    source_type: str
    modality: str
    chunk_index: int
    page_number: Optional[int] = None
    text_preview: Optional[str] = None
    extracted_text: Optional[str] = None
    transcript_text: Optional[str] = None
    llava_description: Optional[str] = None
    original_filename: Optional[str] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    image_index: Optional[int] = None
    image_sha256: Optional[str] = None
    has_text_vector: bool = False
    has_image_vector: bool = False
    linked_records: list[LinkedRecordSchema] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Response — search response
# ---------------------------------------------------------------------------


class SearchResponseSchema(BaseModel):
    """Complete retrieval response with timing metrics and ranked results."""

    query: str
    query_vector_dim: int
    embedding_model: str
    encoding_time_ms: float
    search_time_ms: float
    hydration_time_ms: float
    total_time_ms: float
    result_count: int
    results: list[RetrievalResultSchema] = Field(default_factory=list)
