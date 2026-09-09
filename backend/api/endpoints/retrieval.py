"""Phase 4 retrieval API endpoints.

Provides:
    POST /api/v1/retrieval/search  — semantic search over indexed documents
    GET  /api/v1/retrieval/health  — retrieval subsystem health check
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas.retrieval import (
    LinkedRecordSchema,
    RetrievalResultSchema,
    SearchRequestSchema,
    SearchResponseSchema,
)
from backend.db.database import get_session
from backend.db.models import SourceDocument
from backend.retrieval.retrieval import retrieve
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/search", response_model=SearchResponseSchema)
async def search(request: SearchRequestSchema) -> SearchResponseSchema:
    """Perform semantic search over indexed documents.

    Encodes the query with SentenceTransformer, searches Qdrant text vectors,
    hydrates results from SQLite, and returns ranked evidence with timing.

    This endpoint does NOT generate an LLM answer.
    """
    # Validate that source documents exist and are indexed
    session = get_session()
    try:
        from sqlmodel import select

        stmt = select(SourceDocument).where(
            SourceDocument.source_document_id.in_(request.source_document_ids)  # type: ignore[attr-defined]
        )
        docs = session.exec(stmt).all()
        found_ids = {d.source_document_id for d in docs}
        missing = set(request.source_document_ids) - found_ids
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Source documents not found: {sorted(missing)}",
            )

        # Filter to only indexed documents
        searchable_ids = [
            d.source_document_id
            for d in docs
            if d.indexing_status.value in ("indexed", "completed_with_warnings")
        ]
        if not searchable_ids:
            raise HTTPException(
                status_code=422,
                detail="None of the selected documents are indexed in Qdrant. "
                "Index them in the Knowledge Base first.",
            )
    finally:
        session.close()

    # Execute retrieval
    try:
        response = retrieve(
            query=request.query,
            source_document_ids=searchable_ids,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            modality_filter=request.modality_filter,
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}",
        )

    # Map dataclass response → Pydantic schema
    results = [
        RetrievalResultSchema(
            chunk_id=r.chunk_id,
            score=r.score,
            source_document_id=r.source_document_id,
            source_type=r.source_type,
            modality=r.modality,
            chunk_index=r.chunk_index,
            page_number=r.page_number,
            text_preview=r.text_preview,
            extracted_text=r.extracted_text,
            transcript_text=r.transcript_text,
            llava_description=r.llava_description,
            original_filename=r.original_filename,
            timestamp_start=r.timestamp_start,
            timestamp_end=r.timestamp_end,
            image_index=r.image_index,
            image_sha256=r.image_sha256,
            has_text_vector=r.has_text_vector,
            has_image_vector=r.has_image_vector,
            linked_records=[
                LinkedRecordSchema(
                    chunk_id=lr.chunk_id,
                    source_type=lr.source_type,
                    modality=lr.modality,
                    text_preview=lr.text_preview,
                    llava_description=lr.llava_description,
                )
                for lr in r.linked_records
            ],
        )
        for r in response.results
    ]

    return SearchResponseSchema(
        query=response.query,
        query_vector_dim=response.query_vector_dim,
        embedding_model=response.embedding_model,
        encoding_time_ms=response.encoding_time_ms,
        search_time_ms=response.search_time_ms,
        hydration_time_ms=response.hydration_time_ms,
        total_time_ms=response.total_time_ms,
        result_count=response.result_count,
        results=results,
    )


@router.get("/health")
async def retrieval_health() -> dict:
    """Check retrieval subsystem health.

    Verifies that the TextEmbedder is loadable and Qdrant is reachable.
    """
    from backend.indexing.qdrant_manager import check_qdrant_health

    health: dict = {"subsystem": "retrieval"}

    # Check Qdrant
    qdrant = check_qdrant_health()
    health["qdrant_connected"] = qdrant.get("connected", False)

    # Check TextEmbedder
    try:
        from backend.embeddings.text_embedder import get_text_embedder

        embedder = get_text_embedder()
        health["text_embedder_loaded"] = True
        health["text_embedder_dim"] = embedder.embedding_dim
        health["text_embedder_model"] = embedder.model_name
    except Exception as e:
        health["text_embedder_loaded"] = False
        health["text_embedder_error"] = str(e)

    health["ready"] = health.get("qdrant_connected", False) and health.get(
        "text_embedder_loaded", False
    )
    return health
