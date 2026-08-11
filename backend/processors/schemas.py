"""Shared Pydantic result models for all processors and embedders.

These models are the structured output of Phase 2 modules. They carry
data between processors but are NOT Qdrant points — Qdrant point
construction happens in the indexer (Phase 3).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class ProcessorWarning(BaseModel):
    """Non-fatal warning from a processor (e.g. skipped unreadable image)."""

    processor: str
    message: str
    detail: Optional[str] = None


class ProcessorError(BaseModel):
    """Fatal error from a processor."""

    processor: str
    message: str
    detail: Optional[str] = None
    exception_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class TextChunkResult(BaseModel):
    """A single text chunk produced by the sliding-window chunker."""

    source_document_id: str
    chunk_index: int
    text: str
    token_count: int
    page_number: Optional[int] = None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TextEmbeddingResult(BaseModel):
    """Result of embedding a text string via SentenceTransformer."""

    vector: list[float]
    vector_dimension: int
    embedding_model: str
    embedding_span_count: int
    source_text: str
    token_count: int


class ClipEmbeddingResult(BaseModel):
    """Result of CLIP image embedding."""

    vector: list[float]
    vector_dimension: int
    embedding_model: str
    image_path: str
    preprocessing: str


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


class ExtractedImageResult(BaseModel):
    """An image extracted from a PDF or DOCX document."""

    source_document_id: str
    image_path: str
    image_index: int
    page_number: Optional[int] = None
    image_extension: str
    image_hash: str
    width: Optional[int] = None
    height: Optional[int] = None
    original_filename: str
    source_file_path: str


class PdfPageResult(BaseModel):
    """Text and images extracted from a single PDF page."""

    page_number: int
    text: str
    images: list[ExtractedImageResult] = Field(default_factory=list)


class PdfExtractionResult(BaseModel):
    """Complete extraction result for a PDF file."""

    source_document_id: str
    source_file_path: str
    original_filename: str
    total_pages: int
    pages: list[PdfPageResult]
    extracted_images: list[ExtractedImageResult]
    warnings: list[ProcessorWarning] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------


class DocxTextBlock(BaseModel):
    """A text block from a DOCX file, preserving order and type."""

    block_index: int
    block_type: str  # "heading" | "paragraph" | "table"
    text: str
    heading_level: Optional[int] = None


class DocxExtractionResult(BaseModel):
    """Complete extraction result for a DOCX file."""

    source_document_id: str
    source_file_path: str
    original_filename: str
    text_blocks: list[DocxTextBlock]
    full_text: str
    extracted_images: list[ExtractedImageResult] = Field(default_factory=list)
    warnings: list[ProcessorWarning] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# DOC conversion
# ---------------------------------------------------------------------------


class DocConversionResult(BaseModel):
    """Result of converting a legacy .doc to .docx via LibreOffice."""

    success: bool
    original_path: str
    converted_path: Optional[str] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class AudioNormalizationResult(BaseModel):
    """Result of normalizing audio to WAV 16 kHz mono via ffmpeg."""

    success: bool
    original_path: str
    normalized_path: Optional[str] = None
    original_format: str
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


class AudioSegmentResult(BaseModel):
    """A single transcript segment from faster-whisper."""

    source_document_id: str
    segment_index: int
    start_seconds: float
    end_seconds: float
    transcript_text: str
    language: Optional[str] = None
    model_name: str


class AudioTranscriptionResult(BaseModel):
    """Complete transcription result for an audio file."""

    source_document_id: str
    source_file_path: str
    original_filename: str
    language: Optional[str] = None
    model_name: str
    segments: list[AudioSegmentResult]
    total_duration_seconds: Optional[float] = None
    warnings: list[ProcessorWarning] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


class VisionProcessingResult(BaseModel):
    """Combined CLIP + LLaVA processing result for a single image.

    For every image two Qdrant records will eventually be created:
        A. Image/CLIP record  (image named vector, 512-d)
        B. Description record (text named vector, 384-d) — only if LLaVA succeeds
    This model carries the raw data; Qdrant point construction is Phase 3.
    """

    source_document_id: str
    image_path: str
    original_filename: str
    clip_embedding: Optional[ClipEmbeddingResult] = None
    clip_status: str  # "completed" | "failed"
    llava_description: Optional[str] = None
    llava_description_embedding: Optional[TextEmbeddingResult] = None
    llava_status: str  # "completed" | "unavailable" | "failed"
    ingestion_status: str  # "completed" | "partial_failed" | "failed"
    page_number: Optional[int] = None
    image_hash: Optional[str] = None
    warnings: list[ProcessorWarning] = Field(default_factory=list)
