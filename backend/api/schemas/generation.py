"""Pydantic request/response models for the Phase 5 generation API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ChatMessageSchema(BaseModel):
    """A message in a multi-turn conversation."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")


class AnswerRequestSchema(BaseModel):
    """Request body for POST /api/v1/generation/answer."""

    query: str = Field(..., min_length=1, max_length=2000, description="Natural-language question")
    source_document_ids: list[str] = Field(
        ..., min_length=1, description="Source document IDs to search within"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Max retrieval results")
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum retrieval cosine similarity"
    )
    conversation_history: list[ChatMessageSchema] = Field(
        default_factory=list, description="Recent conversation turns for context"
    )


# ---------------------------------------------------------------------------
# Response — citation
# ---------------------------------------------------------------------------


class CitationSchema(BaseModel):
    """A single citation linking a claim to a source record."""

    source_id: str
    chunk_id: str
    source_document_id: str
    filename: Optional[str] = None
    source_type: str
    modality: str
    page_number: Optional[int] = None
    chunk_index: int
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    score: float
    preview: str


# ---------------------------------------------------------------------------
# Response — retrieval metrics
# ---------------------------------------------------------------------------


class RetrievalMetricsSchema(BaseModel):
    """Phase 4 retrieval timing metrics."""

    encoding_time_ms: float
    search_time_ms: float
    hydration_time_ms: float
    total_time_ms: float
    result_count: int
    embedding_model: str
    query_vector_dim: int


# ---------------------------------------------------------------------------
# Response — generation metrics
# ---------------------------------------------------------------------------


class GenerationMetricsSchema(BaseModel):
    """Phase 5 generation timing and stats."""

    model: str
    generation_time_ms: float
    prompt_eval_count: int
    eval_count: int
    context_record_count: int
    context_char_count: int


# ---------------------------------------------------------------------------
# Response — retrieval result (reused from Phase 4 for evidence cards)
# ---------------------------------------------------------------------------


class LinkedRecordSchema(BaseModel):
    """A cross-modal linked record."""

    chunk_id: str
    source_type: str
    modality: str
    text_preview: Optional[str] = None
    llava_description: Optional[str] = None


class RetrievalResultSchema(BaseModel):
    """A single hydrated retrieval result — for evidence card display."""

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
# Response — full answer
# ---------------------------------------------------------------------------


class AnswerResponseSchema(BaseModel):
    """Complete Phase 5 answer response with citations and metrics."""

    query: str
    answer: str
    grounded: bool
    insufficient_context: bool
    is_conversational: bool = Field(default=False, description="True if response is an interactive conversation turn")
    citations: list[CitationSchema] = Field(default_factory=list)
    retrieval_metrics: Optional[RetrievalMetricsSchema] = None
    generation_metrics: Optional[GenerationMetricsSchema] = None
    warnings: list[str] = Field(default_factory=list)
    retrieval_results: list[RetrievalResultSchema] = Field(default_factory=list)
