"""Phase 3 Indexer — embed, upsert to Qdrant, and persist IndexedRecords.

This module is the Phase 3 engine.  It takes processor outputs
(text chunks, vision results, audio segments) and:
    1. Generates deterministic UUIDv5 point IDs
    2. Embeds content into the correct vector space
    3. Upserts Qdrant points with structured payloads
    4. Persists IndexedRecord rows in SQLite
    5. Maintains cross-modal links for image records

Scope boundary: This module does NOT implement query encoding, search,
retrieval, ranking, or answer generation (Phase 4+).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from qdrant_client import models

from backend.config import settings
from backend.db.database import get_session
from backend.db.models import (
    IndexedRecord,
    IndexingStatus,
    JobStatus,
    Modality,
    RecordSourceType,
    SourceDocument,
)
from backend.embeddings.text_embedder import get_text_embedder
from backend.indexing.qdrant_manager import (
    count_points_by_source,
    delete_points_by_source,
    upsert_points_batch,
)
from backend.processors.schemas import (
    AudioSegmentResult,
    TextChunkResult,
    VisionProcessingResult,
)
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Deterministic UUID generation (UUIDv5)
# ---------------------------------------------------------------------------

# Fixed namespace UUID for the RAG system — stable across restarts
_RAG_NAMESPACE = uuid.UUID("d7e5b9a2-4c3f-4e8a-b1d6-9f2a3c5e7d8b")

# Max chars of extracted text stored in Qdrant payload as preview
_PAYLOAD_TEXT_PREVIEW_LIMIT = 500


def make_text_chunk_id(source_document_id: str, chunk_index: int) -> str:
    """Deterministic ID for a text chunk record."""
    return str(uuid.uuid5(
        _RAG_NAMESPACE,
        f"{source_document_id}:text_chunk:{chunk_index}",
    ))


def make_clip_id(
    source_document_id: str,
    image_hash: str,
    page_number: int | None = None,
    image_index: int = 0,
) -> str:
    """Deterministic ID for a CLIP image record.

    Includes page_number and image_index so that the same image asset
    appearing multiple times in a document (e.g. a repeated logo) gets
    a unique but stable ID per occurrence.
    """
    page = page_number if page_number is not None else 0
    return str(uuid.uuid5(
        _RAG_NAMESPACE,
        f"{source_document_id}:clip:page:{page}:image:{image_index}:sha256:{image_hash}",
    ))


def make_llava_desc_id(
    source_document_id: str,
    image_hash: str,
    page_number: int | None = None,
    image_index: int = 0,
) -> str:
    """Deterministic ID for a LLaVA description record.

    Mirrors make_clip_id with the same occurrence-awareness.
    """
    page = page_number if page_number is not None else 0
    return str(uuid.uuid5(
        _RAG_NAMESPACE,
        f"{source_document_id}:llava_desc:page:{page}:image:{image_index}:sha256:{image_hash}",
    ))


def make_audio_segment_id(
    source_document_id: str,
    segment_index: int,
    start: float,
    end: float,
) -> str:
    """Deterministic ID for an audio segment record."""
    return str(uuid.uuid5(
        _RAG_NAMESPACE,
        f"{source_document_id}:audio:{segment_index}:{start:.3f}-{end:.3f}",
    ))


# ---------------------------------------------------------------------------
# Qdrant payload builder
# ---------------------------------------------------------------------------


def _build_payload(
    *,
    source_document_id: str,
    source_type: str,
    modality: str,
    chunk_index: int,
    original_filename: str,
    embedding_model: str,
    text_preview: str = "",
    page_number: int | None = None,
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
    linked_chunk_ids: list[str] | None = None,
    indexing_status: str = "completed",
    image_hash: str | None = None,
) -> dict:
    """Build a structured Qdrant point payload."""
    payload = {
        "source_document_id": source_document_id,
        "source_type": source_type,
        "modality": modality,
        "chunk_index": chunk_index,
        "page_number": page_number,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "indexing_status": indexing_status,
        "text_preview": text_preview[:_PAYLOAD_TEXT_PREVIEW_LIMIT],
        "linked_chunk_ids": linked_chunk_ids or [],
        "original_filename": original_filename,
        "embedding_model": embedding_model,
    }
    if image_hash:
        payload["image_hash"] = image_hash
    return payload


# ---------------------------------------------------------------------------
# Text chunk indexing
# ---------------------------------------------------------------------------


def index_text_chunks(
    chunks: list[TextChunkResult],
    source_document_id: str,
    source_type: RecordSourceType,
    original_filename: str,
) -> dict:
    """Embed text chunks, upsert to Qdrant, and persist IndexedRecords.

    Returns a summary dict with counts and any warnings.
    """
    if not chunks:
        return {"indexed": 0, "warnings": []}

    text_embedder = get_text_embedder()
    qdrant_points: list[models.PointStruct] = []
    db_records: list[IndexedRecord] = []
    warnings: list[str] = []

    for chunk in chunks:
        chunk_id = make_text_chunk_id(source_document_id, chunk.chunk_index)

        try:
            emb_result = text_embedder.embed(chunk.text)
        except Exception as exc:
            warnings.append(
                f"Embedding failed for chunk {chunk.chunk_index}: {exc}"
            )
            logger.warning("Embedding failed for chunk %d: %s", chunk.chunk_index, exc)
            continue

        payload = _build_payload(
            source_document_id=source_document_id,
            source_type=source_type.value,
            modality=Modality.text.value,
            chunk_index=chunk.chunk_index,
            original_filename=original_filename,
            embedding_model=emb_result.embedding_model,
            text_preview=chunk.text,
            page_number=chunk.page_number,
        )

        qdrant_points.append(models.PointStruct(
            id=chunk_id,
            vector={"text": emb_result.vector},
            payload=payload,
        ))

        db_records.append(IndexedRecord(
            chunk_id=chunk_id,
            qdrant_point_id=chunk_id,
            source_document_id=source_document_id,
            source_type=source_type,
            modality=Modality.text,
            chunk_index=chunk.chunk_index,
            extracted_text=chunk.text,  # Full text in SQLite
            page_number=chunk.page_number,
            has_text_vector=True,
            has_image_vector=False,
            embedding_span_count=emb_result.embedding_span_count,
            embedding_model=emb_result.embedding_model,
            ingestion_status=JobStatus.completed,
            created_at=datetime.now(timezone.utc),
        ))

    # Batch upsert to Qdrant
    upserted = upsert_points_batch(qdrant_points)

    # Persist to SQLite
    with get_session() as session:
        for rec in db_records:
            session.add(rec)
        session.commit()

    logger.info(
        "Indexed %d text chunks for %s (source_type=%s, warnings=%d)",
        upserted, source_document_id, source_type.value, len(warnings),
    )
    return {"indexed": upserted, "warnings": warnings}


# ---------------------------------------------------------------------------
# Image indexing (dual CLIP + LLaVA records)
# ---------------------------------------------------------------------------


def index_image(
    vision_result: VisionProcessingResult,
    source_document_id: str,
    clip_source_type: RecordSourceType,
    desc_source_type: RecordSourceType,
    original_filename: str,
    image_index: int = 0,
) -> dict:
    """Create Qdrant points and IndexedRecords for an image.

    Creates up to two linked records:
        A. CLIP image record (512-d image vector)
        B. LLaVA description record (384-d text vector)

    If LLaVA failed, only Record A is created (partial success).

    IDs are occurrence-aware: they include page_number and image_index
    so that the same image asset repeated across pages produces unique
    but deterministic point IDs.

    Returns a summary dict.
    """
    image_hash = vision_result.image_hash or "unknown"
    page_num = vision_result.page_number

    # Occurrence-aware deterministic IDs
    clip_id = make_clip_id(
        source_document_id, image_hash,
        page_number=page_num, image_index=image_index,
    )
    desc_id = make_llava_desc_id(
        source_document_id, image_hash,
        page_number=page_num, image_index=image_index,
    )

    qdrant_points: list[models.PointStruct] = []
    db_records: list[IndexedRecord] = []
    warnings: list[str] = []
    record_status = "completed"

    # --- Record A: CLIP image vector ---
    if vision_result.clip_embedding:
        linked_ids = [desc_id] if vision_result.llava_description_embedding else []

        payload_a = _build_payload(
            source_document_id=source_document_id,
            source_type=clip_source_type.value,
            modality=Modality.image.value,
            chunk_index=image_index,
            original_filename=original_filename,
            embedding_model=vision_result.clip_embedding.embedding_model,
            text_preview="",
            page_number=page_num,
            linked_chunk_ids=linked_ids,
            image_hash=image_hash,
        )
        payload_a["image_index"] = image_index

        qdrant_points.append(models.PointStruct(
            id=clip_id,
            vector={"image": vision_result.clip_embedding.vector},
            payload=payload_a,
        ))

        db_records.append(IndexedRecord(
            chunk_id=clip_id,
            qdrant_point_id=clip_id,
            source_document_id=source_document_id,
            source_type=clip_source_type,
            modality=Modality.image,
            chunk_index=image_index,
            page_number=page_num,
            image_index=image_index,
            image_sha256=image_hash,
            has_text_vector=False,
            has_image_vector=True,
            embedding_model=vision_result.clip_embedding.embedding_model,
            linked_chunk_ids_json=json.dumps(linked_ids),
            llava_status=vision_result.llava_status,
            ingestion_status=JobStatus.completed,
            created_at=datetime.now(timezone.utc),
        ))
    else:
        warnings.append(f"CLIP embedding failed for image {original_filename}")
        return {"indexed": 0, "warnings": warnings, "status": "failed"}

    # --- Record B: LLaVA description text vector ---
    if vision_result.llava_description_embedding and vision_result.llava_description:
        payload_b = _build_payload(
            source_document_id=source_document_id,
            source_type=desc_source_type.value,
            modality=Modality.text.value,
            chunk_index=image_index,
            original_filename=original_filename,
            embedding_model=vision_result.llava_description_embedding.embedding_model,
            text_preview=vision_result.llava_description,
            page_number=page_num,
            linked_chunk_ids=[clip_id],
            image_hash=image_hash,
        )
        payload_b["image_index"] = image_index

        qdrant_points.append(models.PointStruct(
            id=desc_id,
            vector={"text": vision_result.llava_description_embedding.vector},
            payload=payload_b,
        ))

        db_records.append(IndexedRecord(
            chunk_id=desc_id,
            qdrant_point_id=desc_id,
            source_document_id=source_document_id,
            source_type=desc_source_type,
            modality=Modality.text,
            chunk_index=image_index,
            llava_description=vision_result.llava_description,  # Full in SQLite
            page_number=page_num,
            image_index=image_index,
            image_sha256=image_hash,
            has_text_vector=True,
            has_image_vector=False,
            embedding_model=vision_result.llava_description_embedding.embedding_model,
            linked_chunk_ids_json=json.dumps([clip_id]),
            llava_status=vision_result.llava_status,
            ingestion_status=JobStatus.completed,
            created_at=datetime.now(timezone.utc),
        ))
    else:
        record_status = "completed_with_warnings"
        warnings.append(
            f"LLaVA description unavailable for {original_filename}; "
            "only CLIP record indexed"
        )

    # Upsert + persist
    upserted = upsert_points_batch(qdrant_points)

    with get_session() as session:
        try:
            for rec in db_records:
                session.add(rec)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(
                "SQLite commit failed for image %s (idx=%d, page=%s, hash=%s): %s",
                original_filename, image_index, page_num, image_hash, exc,
            )
            raise

    logger.info(
        "Indexed image %s: %d points (idx=%d, page=%s, hash=%s, CLIP=%s, LLaVA=%s)",
        original_filename, upserted, image_index, page_num, image_hash,
        vision_result.clip_status, vision_result.llava_status,
    )
    return {"indexed": upserted, "warnings": warnings, "status": record_status}


# ---------------------------------------------------------------------------
# Audio segment indexing
# ---------------------------------------------------------------------------


def index_audio_segments(
    segments: list[AudioSegmentResult],
    source_document_id: str,
    original_filename: str,
) -> dict:
    """Embed audio transcript segments, upsert to Qdrant, persist records.

    Returns a summary dict.
    """
    if not segments:
        return {"indexed": 0, "warnings": []}

    text_embedder = get_text_embedder()
    qdrant_points: list[models.PointStruct] = []
    db_records: list[IndexedRecord] = []
    warnings: list[str] = []

    for seg in segments:
        seg_id = make_audio_segment_id(
            source_document_id,
            seg.segment_index,
            seg.start_seconds,
            seg.end_seconds,
        )

        transcript = getattr(seg, "transcript_text", "") or ""
        if not transcript.strip():
            warnings.append(f"Empty transcript at segment {seg.segment_index}")
            continue

        try:
            emb_result = text_embedder.embed(transcript)
        except Exception as exc:
            warnings.append(
                f"Embedding failed for audio segment {seg.segment_index}: {exc}"
            )
            logger.warning(
                "Audio segment %d embedding failed: %s",
                seg.segment_index, exc,
            )
            continue

        payload = _build_payload(
            source_document_id=source_document_id,
            source_type=RecordSourceType.audio.value,
            modality=Modality.text.value,  # Audio transcripts use text vector
            chunk_index=seg.segment_index,
            original_filename=original_filename,
            embedding_model=emb_result.embedding_model,
            text_preview=transcript,
            timestamp_start=seg.start_seconds,
            timestamp_end=seg.end_seconds,
        )

        qdrant_points.append(models.PointStruct(
            id=seg_id,
            vector={"text": emb_result.vector},
            payload=payload,
        ))

        db_records.append(IndexedRecord(
            chunk_id=seg_id,
            qdrant_point_id=seg_id,
            source_document_id=source_document_id,
            source_type=RecordSourceType.audio,
            modality=Modality.text,  # Stored in text vector space
            chunk_index=seg.segment_index,
            transcript_text=transcript,  # Full transcript in SQLite
            timestamp_start=seg.start_seconds,
            timestamp_end=seg.end_seconds,
            has_text_vector=True,
            has_image_vector=False,
            embedding_span_count=emb_result.embedding_span_count,
            embedding_model=emb_result.embedding_model,
            ingestion_status=JobStatus.completed,
            created_at=datetime.now(timezone.utc),
        ))

    upserted = upsert_points_batch(qdrant_points)

    with get_session() as session:
        for rec in db_records:
            session.add(rec)
        session.commit()

    logger.info(
        "Indexed %d audio segments for %s (warnings=%d)",
        upserted, source_document_id, len(warnings),
    )
    return {"indexed": upserted, "warnings": warnings}


# ---------------------------------------------------------------------------
# Safe deletion (Qdrant + SQLite)
# ---------------------------------------------------------------------------


def delete_indexed_records(source_document_id: str) -> dict:
    """Delete all Qdrant points and SQLite IndexedRecords for a document.

    Steps:
        1. Delete Qdrant points by source_document_id (wait=True)
        2. Verify Qdrant point count is zero
        3. Delete IndexedRecord rows from SQLite

    Returns a summary dict.
    """
    # Step 1: Delete from Qdrant
    try:
        delete_points_by_source(source_document_id, wait=True)
    except Exception as exc:
        logger.error(
            "Qdrant deletion failed for %s: %s", source_document_id, exc
        )
        return {"deleted_qdrant": 0, "deleted_sqlite": 0, "error": str(exc)}

    # Step 2: Verify deletion
    try:
        remaining = count_points_by_source(source_document_id)
        if remaining > 0:
            logger.warning(
                "Qdrant still has %d points for %s after deletion",
                remaining, source_document_id,
            )
    except Exception as exc:
        logger.warning("Could not verify Qdrant deletion: %s", exc)

    # Step 3: Delete from SQLite
    deleted_count = 0
    with get_session() as session:
        from sqlmodel import select

        records = session.exec(
            select(IndexedRecord).where(
                IndexedRecord.source_document_id == source_document_id
            )
        ).all()
        deleted_count = len(records)
        for rec in records:
            session.delete(rec)
        session.commit()

    logger.info(
        "Deleted indexed records for %s: %d SQLite rows",
        source_document_id, deleted_count,
    )
    return {"deleted_qdrant": True, "deleted_sqlite": deleted_count}


# ---------------------------------------------------------------------------
# Index status / stats
# ---------------------------------------------------------------------------


def get_index_status(source_document_id: str) -> dict:
    """Return indexing statistics for a source document.

    Returns point counts, modality breakdown, and linked pairs.
    """
    with get_session() as session:
        from sqlmodel import select

        records = session.exec(
            select(IndexedRecord).where(
                IndexedRecord.source_document_id == source_document_id
            )
        ).all()

    total = len(records)
    by_modality: dict[str, int] = {}
    by_source_type: dict[str, int] = {}
    linked_pairs = 0

    for rec in records:
        mod = rec.modality.value if rec.modality else "unknown"
        by_modality[mod] = by_modality.get(mod, 0) + 1

        st = rec.source_type.value if rec.source_type else "unknown"
        by_source_type[st] = by_source_type.get(st, 0) + 1

        if rec.linked_chunk_ids_json:
            ids = json.loads(rec.linked_chunk_ids_json)
            if ids:
                linked_pairs += 1

    # Also check live Qdrant count
    qdrant_count = 0
    try:
        qdrant_count = count_points_by_source(source_document_id)
    except Exception as exc:
        logger.warning("Could not get Qdrant count for %s: %s", source_document_id, exc)

    return {
        "source_document_id": source_document_id,
        "total_indexed_records": total,
        "qdrant_point_count": qdrant_count,
        "by_modality": by_modality,
        "by_source_type": by_source_type,
        "linked_pairs": linked_pairs // 2,  # Each pair counted twice
    }
