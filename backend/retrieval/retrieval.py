"""Phase 4 retrieval engine.

Pipeline:
    1. Encode user query → 384-d text vector via TextEmbedder
    2. Search Qdrant ``text`` named-vector space with payload filters
    3. Hydrate matched records from SQLite (IndexedRecord + SourceDocument)
    4. Resolve cross-modal ``linked_chunk_ids`` for enrichment
    5. Return ranked, traceable evidence with timing metrics
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from sqlmodel import select

from backend.config import settings
from backend.db.database import get_session
from backend.db.models import IndexedRecord, SourceDocument
from backend.embeddings.text_embedder import get_text_embedder
from backend.indexing.search import search_text_vectors
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LinkedRecord:
    """Compact representation of a cross-modal linked record."""

    chunk_id: str
    source_type: str
    modality: str
    text_preview: str | None = None
    llava_description: str | None = None


@dataclass
class RetrievalResult:
    """A single hydrated retrieval result with full metadata."""

    chunk_id: str
    score: float
    source_document_id: str
    source_type: str
    modality: str
    chunk_index: int
    page_number: int | None = None
    text_preview: str | None = None
    extracted_text: str | None = None
    transcript_text: str | None = None
    llava_description: str | None = None
    original_filename: str | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    image_index: int | None = None
    image_sha256: str | None = None
    has_text_vector: bool = False
    has_image_vector: bool = False
    linked_records: list[LinkedRecord] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Complete retrieval response including timing metrics."""

    query: str
    query_vector_dim: int
    embedding_model: str
    encoding_time_ms: float
    search_time_ms: float
    hydration_time_ms: float
    total_time_ms: float
    result_count: int
    results: list[RetrievalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    source_document_ids: list[str],
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    modality_filter: str | None = None,
) -> SearchResponse:
    """Execute the full retrieval pipeline.

    Parameters
    ----------
    query:
        Natural-language user query.
    source_document_ids:
        List of source document IDs to search within.
    top_k:
        Maximum results.  Defaults to ``settings.retrieval_top_k``.
    score_threshold:
        Minimum cosine score.  Defaults to ``settings.retrieval_score_threshold``.
    modality_filter:
        Optional ``"text"`` | ``"image"`` | ``"audio"`` filter.

    Returns
    -------
    SearchResponse
        Ranked results with full hydration and timing.
    """
    effective_top_k = top_k if top_k is not None else settings.retrieval_top_k
    effective_threshold = (
        score_threshold
        if score_threshold is not None
        else settings.retrieval_score_threshold
    )

    t_total_start = time.perf_counter()

    # ---- Step 1: Encode query ----
    t_enc_start = time.perf_counter()
    embedder = get_text_embedder()
    embed_result = embedder.embed(query)
    query_vector = embed_result.vector
    t_enc_end = time.perf_counter()
    encoding_time_ms = (t_enc_end - t_enc_start) * 1000

    logger.info(
        "Query encoded: dim=%d, model=%s, time=%.1fms",
        embed_result.vector_dimension,
        embed_result.embedding_model,
        encoding_time_ms,
    )

    # ---- Step 2: Qdrant search ----
    t_search_start = time.perf_counter()
    scored_points = search_text_vectors(
        query_vector=query_vector,
        source_document_ids=source_document_ids,
        limit=effective_top_k,
        score_threshold=effective_threshold,
        modality_filter=modality_filter,
    )
    t_search_end = time.perf_counter()
    search_time_ms = (t_search_end - t_search_start) * 1000

    if not scored_points:
        t_total_end = time.perf_counter()
        return SearchResponse(
            query=query,
            query_vector_dim=embed_result.vector_dimension,
            embedding_model=embed_result.embedding_model,
            encoding_time_ms=round(encoding_time_ms, 1),
            search_time_ms=round(search_time_ms, 1),
            hydration_time_ms=0.0,
            total_time_ms=round((t_total_end - t_total_start) * 1000, 1),
            result_count=0,
            results=[],
        )

    # ---- Step 3: SQLite hydration ----
    t_hydrate_start = time.perf_counter()

    # Map point_id → score and preserve ordering
    point_id_to_score: dict[str, float] = {}
    point_id_order: list[str] = []
    for sp in scored_points:
        pid = str(sp.id)
        point_id_to_score[pid] = sp.score
        point_id_order.append(pid)

    session = get_session()
    try:
        # Fetch matching IndexedRecords
        stmt = select(IndexedRecord).where(
            IndexedRecord.qdrant_point_id.in_(point_id_order)  # type: ignore[attr-defined]
        )
        records = session.exec(stmt).all()
        record_map: dict[str, IndexedRecord] = {
            r.qdrant_point_id: r for r in records
        }

        # Fetch SourceDocuments for filenames
        doc_ids = list({r.source_document_id for r in records})
        doc_stmt = select(SourceDocument).where(
            SourceDocument.source_document_id.in_(doc_ids)  # type: ignore[attr-defined]
        )
        docs = session.exec(doc_stmt).all()
        doc_map: dict[str, SourceDocument] = {
            d.source_document_id: d for d in docs
        }

        # Collect all linked_chunk_ids for batch fetch
        all_linked_ids: set[str] = set()
        for rec in records:
            if rec.linked_chunk_ids_json:
                try:
                    linked = json.loads(rec.linked_chunk_ids_json)
                    all_linked_ids.update(linked)
                except (json.JSONDecodeError, TypeError):
                    pass

        linked_map: dict[str, IndexedRecord] = {}
        if all_linked_ids:
            linked_stmt = select(IndexedRecord).where(
                IndexedRecord.chunk_id.in_(list(all_linked_ids))  # type: ignore[attr-defined]
            )
            linked_records = session.exec(linked_stmt).all()
            linked_map = {lr.chunk_id: lr for lr in linked_records}

        # ---- Step 4: Build results in score order ----
        results: list[RetrievalResult] = []
        for pid in point_id_order:
            rec = record_map.get(pid)
            if rec is None:
                # Point exists in Qdrant but not in SQLite — skip
                logger.warning("Qdrant point %s not found in SQLite — skipping", pid)
                continue

            doc = doc_map.get(rec.source_document_id)

            # Resolve payload text_preview from Qdrant point
            sp_match = next((sp for sp in scored_points if str(sp.id) == pid), None)
            text_preview = None
            if sp_match and sp_match.payload:
                text_preview = sp_match.payload.get("text_preview")

            # Resolve linked records
            linked_results: list[LinkedRecord] = []
            if rec.linked_chunk_ids_json:
                try:
                    linked_ids = json.loads(rec.linked_chunk_ids_json)
                    for lid in linked_ids:
                        lr = linked_map.get(lid)
                        if lr:
                            # Get text preview for linked record
                            lr_preview = None
                            if lr.extracted_text:
                                lr_preview = lr.extracted_text[:200]
                            elif lr.transcript_text:
                                lr_preview = lr.transcript_text[:200]

                            linked_results.append(
                                LinkedRecord(
                                    chunk_id=lr.chunk_id,
                                    source_type=lr.source_type.value if hasattr(lr.source_type, 'value') else str(lr.source_type),
                                    modality=lr.modality.value if hasattr(lr.modality, 'value') else str(lr.modality),
                                    text_preview=lr_preview,
                                    llava_description=lr.llava_description,
                                )
                            )
                except (json.JSONDecodeError, TypeError):
                    pass

            results.append(
                RetrievalResult(
                    chunk_id=rec.chunk_id,
                    score=round(point_id_to_score.get(pid, 0.0), 4),
                    source_document_id=rec.source_document_id,
                    source_type=rec.source_type.value if hasattr(rec.source_type, 'value') else str(rec.source_type),
                    modality=rec.modality.value if hasattr(rec.modality, 'value') else str(rec.modality),
                    chunk_index=rec.chunk_index,
                    page_number=rec.page_number,
                    text_preview=text_preview,
                    extracted_text=rec.extracted_text,
                    transcript_text=rec.transcript_text,
                    llava_description=rec.llava_description,
                    original_filename=doc.original_filename if doc else None,
                    timestamp_start=rec.timestamp_start,
                    timestamp_end=rec.timestamp_end,
                    image_index=rec.image_index,
                    image_sha256=rec.image_sha256,
                    has_text_vector=rec.has_text_vector,
                    has_image_vector=rec.has_image_vector,
                    linked_records=linked_results,
                )
            )

    finally:
        session.close()

    t_hydrate_end = time.perf_counter()
    hydration_time_ms = (t_hydrate_end - t_hydrate_start) * 1000

    t_total_end = time.perf_counter()
    total_time_ms = (t_total_end - t_total_start) * 1000

    logger.info(
        "Retrieval complete: %d results in %.1fms "
        "(encode=%.1fms, search=%.1fms, hydrate=%.1fms)",
        len(results),
        total_time_ms,
        encoding_time_ms,
        search_time_ms,
        hydration_time_ms,
    )

    return SearchResponse(
        query=query,
        query_vector_dim=embed_result.vector_dimension,
        embedding_model=embed_result.embedding_model,
        encoding_time_ms=round(encoding_time_ms, 1),
        search_time_ms=round(search_time_ms, 1),
        hydration_time_ms=round(hydration_time_ms, 1),
        total_time_ms=round(total_time_ms, 1),
        result_count=len(results),
        results=results,
    )
