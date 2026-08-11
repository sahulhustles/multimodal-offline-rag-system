"""SQLModel ORM models for ingestion tracking and record management.

Identifier conventions (mandated by architecture):
    source_document_id  — Identifies the original uploaded source. Shared by all
                          chunks, extracted images, descriptions, and audio segments
                          originating from that source.
    chunk_id            — Unique UUID for every logical indexed record (text chunk,
                          CLIP image record, LLaVA-description record, audio segment).
    qdrant_point_id     — Stable UUID used as the Qdrant point ID. Set equal to
                          chunk_id unless there is a technical reason to differ.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import SQLModel, Field, Relationship


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Status of an ingestion job or individual record."""

    queued = "queued"
    processing = "processing"
    completed = "completed"
    partial_failed = "partial_failed"
    failed = "failed"


class StepStatus(str, Enum):
    """Status of a single processing step within a job."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class DocumentSourceType(str, Enum):
    """Source types for SourceDocument — only top-level uploaded types."""

    pdf = "pdf"
    doc = "doc"
    docx = "docx"
    image = "image"
    screenshot = "screenshot"
    audio = "audio"
    text_note = "text_note"


class RecordSourceType(str, Enum):
    """Source types for IndexedRecord — includes all derived record types.

    Top-level:
        pdf, doc, docx, audio, text_note — text chunks from these sources.
    Image CLIP records:
        image, screenshot, pdf_extracted_image — CLIP 512-d vector only.
    Image description records:
        image_description, screenshot_description, pdf_image_description
        — SentenceTransformer 384-d vector of LLaVA description only.
    """

    # Document text chunks
    pdf = "pdf"
    doc = "doc"
    docx = "docx"
    text_note = "text_note"

    # Audio transcript segments
    audio = "audio"

    # Image CLIP records (image named vector only)
    image = "image"
    screenshot = "screenshot"
    pdf_extracted_image = "pdf_extracted_image"

    # Image description records (text named vector only)
    image_description = "image_description"
    screenshot_description = "screenshot_description"
    pdf_image_description = "pdf_image_description"


class Modality(str, Enum):
    """Content modality of an indexed record."""

    text = "text"
    image = "image"
    audio = "audio"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class IngestionJob(SQLModel, table=True):
    """Tracks a single ingestion job triggered by an upload or text-note submission.

    One job may produce one or more SourceDocuments (e.g. multi-file upload).
    """

    __tablename__ = "ingestion_jobs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    status: JobStatus = Field(default=JobStatus.queued, index=True)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    mock_mode: bool = Field(default=False)

    # Relationships
    processing_steps: list["ProcessingStep"] = Relationship(back_populates="job")
    source_documents: list["SourceDocument"] = Relationship(back_populates="job")


class ProcessingStep(SQLModel, table=True):
    """A single processing step within an ingestion job's timeline.

    Steps include: file_validation, text_extraction, image_extraction,
    chunking, text_embedding, vision_processing, audio_conversion,
    audio_transcription, indexing, etc.
    """

    __tablename__ = "processing_steps"

    id: str = Field(default_factory=_uuid, primary_key=True)
    job_id: str = Field(foreign_key="ingestion_jobs.id", index=True)
    step_name: str = Field(index=True)
    status: StepStatus = Field(default=StepStatus.pending)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    duration_ms: Optional[int] = Field(default=None)
    detail: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)

    # Relationships
    job: Optional[IngestionJob] = Relationship(back_populates="processing_steps")


class SourceDocument(SQLModel, table=True):
    """Represents an original uploaded source file or text note.

    All chunks, extracted images, image descriptions, and audio segments
    derived from this source share the same source_document_id.
    """

    __tablename__ = "source_documents"

    source_document_id: str = Field(default_factory=_uuid, primary_key=True)
    job_id: str = Field(foreign_key="ingestion_jobs.id", index=True)
    original_filename: str
    source_type: DocumentSourceType
    file_path: Optional[str] = Field(default=None)
    file_hash: Optional[str] = Field(default=None)
    file_size_bytes: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)
    ingestion_status: JobStatus = Field(default=JobStatus.queued, index=True)
    error_message: Optional[str] = Field(default=None)

    # Relationships
    job: Optional[IngestionJob] = Relationship(back_populates="source_documents")
    indexed_records: list["IndexedRecord"] = Relationship(back_populates="source_document")


class IndexedRecord(SQLModel, table=True):
    """A single indexed record stored as a Qdrant point.

    Each record has a unique chunk_id and corresponds to exactly one
    Qdrant point (qdrant_point_id == chunk_id by default).

    For images, there are TWO linked records per image:
        A. Image/CLIP record  — modality='image', has image named vector (512d)
        B. Description record — modality='text',  has text named vector (384d)
    Both share the same source_document_id and link to each other
    via linked_chunk_ids.
    """

    __tablename__ = "indexed_records"

    chunk_id: str = Field(default_factory=_uuid, primary_key=True)
    qdrant_point_id: str = Field(index=True)
    source_document_id: str = Field(
        foreign_key="source_documents.source_document_id", index=True
    )
    source_type: RecordSourceType
    modality: Modality
    chunk_index: int = Field(default=0)

    # Content fields (mutually exclusive depending on record type)
    extracted_text: Optional[str] = Field(default=None)
    transcript_text: Optional[str] = Field(default=None)
    llava_description: Optional[str] = Field(default=None)
    llava_status: Optional[str] = Field(default=None)

    # Location metadata
    page_number: Optional[int] = Field(default=None)
    timestamp_start: Optional[float] = Field(default=None)
    timestamp_end: Optional[float] = Field(default=None)

    # Vector presence tracking
    has_text_vector: bool = Field(default=False)
    has_image_vector: bool = Field(default=False)

    # Embedding metadata
    embedding_span_count: Optional[int] = Field(default=None)
    embedding_model: Optional[str] = Field(default=None)

    # Cross-modal linking: JSON-serialized list of chunk_id strings
    linked_chunk_ids_json: Optional[str] = Field(default=None)

    # Status
    ingestion_status: JobStatus = Field(default=JobStatus.queued, index=True)
    created_at: datetime = Field(default_factory=_utcnow)
    error_message: Optional[str] = Field(default=None)
    mock_mode: bool = Field(default=False)

    # DOC conversion metadata (only for doc source_type)
    doc_conversion_status: Optional[str] = Field(default=None)
    doc_conversion_error: Optional[str] = Field(default=None)

    # Audio metadata
    original_format: Optional[str] = Field(default=None)
    normalized_audio_path: Optional[str] = Field(default=None)
    audio_duration_seconds: Optional[float] = Field(default=None)

    # Relationships
    source_document: Optional[SourceDocument] = Relationship(
        back_populates="indexed_records"
    )

    # --- Convenience methods ---

    @property
    def linked_chunk_ids(self) -> list[str]:
        """Deserialise linked_chunk_ids from JSON string."""
        if not self.linked_chunk_ids_json:
            return []
        import json
        return json.loads(self.linked_chunk_ids_json)

    @linked_chunk_ids.setter
    def linked_chunk_ids(self, value: list[str]) -> None:
        """Serialise linked_chunk_ids to JSON string for SQLite storage."""
        import json
        self.linked_chunk_ids_json = json.dumps(value)
