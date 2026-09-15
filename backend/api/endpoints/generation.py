"""Phase 5 generation API endpoints.

Provides:
    POST /api/v1/generation/answer  — grounded answer generation
    GET  /api/v1/generation/health  — generation subsystem health check
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas.generation import (
    AnswerRequestSchema,
    AnswerResponseSchema,
    CitationSchema,
    GenerationMetricsSchema,
    LinkedRecordSchema,
    RetrievalMetricsSchema,
    RetrievalResultSchema,
)
from backend.core.exceptions import ModelUnavailableError
from backend.db.database import get_session
from backend.db.models import SourceDocument
from backend.generation.generator import generate_answer
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/answer", response_model=AnswerResponseSchema)
async def answer(request: AnswerRequestSchema) -> AnswerResponseSchema:
    """Generate a grounded answer from retrieved evidence.

    Calls Phase 4 retrieval, assembles context, generates via Ollama,
    and validates citations.  Returns the answer only if citation
    validation passes; otherwise returns a safe failure state.
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

    # Execute generation pipeline
    try:
        history_dicts = [m.model_dump() for m in request.conversation_history]
        response = generate_answer(
            query=request.query,
            source_document_ids=searchable_ids,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            conversation_history=history_dicts,
        )
    except ModelUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Generation model unavailable: {str(e)}",
        )
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )

    # Map dataclass response → Pydantic schema
    citations = [
        CitationSchema(
            source_id=c.source_id,
            chunk_id=c.chunk_id,
            source_document_id=c.source_document_id,
            filename=c.filename,
            source_type=c.source_type,
            modality=c.modality,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            timestamp_start=c.timestamp_start,
            timestamp_end=c.timestamp_end,
            score=c.score,
            preview=c.preview,
        )
        for c in response.citations
    ]

    retrieval_metrics = None
    if response.retrieval_metrics:
        retrieval_metrics = RetrievalMetricsSchema(
            encoding_time_ms=response.retrieval_metrics.encoding_time_ms,
            search_time_ms=response.retrieval_metrics.search_time_ms,
            hydration_time_ms=response.retrieval_metrics.hydration_time_ms,
            total_time_ms=response.retrieval_metrics.total_time_ms,
            result_count=response.retrieval_metrics.result_count,
            embedding_model=response.retrieval_metrics.embedding_model,
            query_vector_dim=response.retrieval_metrics.query_vector_dim,
        )

    generation_metrics = None
    if response.generation_metrics:
        generation_metrics = GenerationMetricsSchema(
            model=response.generation_metrics.model,
            generation_time_ms=response.generation_metrics.generation_time_ms,
            prompt_eval_count=response.generation_metrics.prompt_eval_count,
            eval_count=response.generation_metrics.eval_count,
            context_record_count=response.generation_metrics.context_record_count,
            context_char_count=response.generation_metrics.context_char_count,
        )

    retrieval_results = [
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
        for r in response.retrieval_results
    ]

    return AnswerResponseSchema(
        query=response.query,
        answer=response.answer,
        grounded=response.grounded,
        insufficient_context=response.insufficient_context,
        is_conversational=response.is_conversational,
        citations=citations,
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        warnings=response.warnings,
        retrieval_results=retrieval_results,
    )


@router.get("/health")
async def generation_health() -> dict:
    """Check generation subsystem health.

    Verifies that the Ollama generation model is available and reachable.
    """
    from backend.config import settings
    from backend.generation.ollama_chat import get_chat_client

    health: dict = {"subsystem": "generation"}

    # Check Ollama generation model
    try:
        client = get_chat_client()
        available = client.is_available()
        health["ollama_reachable"] = available
        health["generation_model"] = settings.generation_model
        health["generation_temperature"] = settings.generation_temperature
        health["generation_max_tokens"] = settings.generation_max_tokens
        health["generation_min_score"] = settings.generation_min_score
    except Exception as e:
        health["ollama_reachable"] = False
        health["ollama_error"] = str(e)

    # Check retrieval subsystem (needed as prereq)
    try:
        from backend.embeddings.text_embedder import get_text_embedder
        from backend.indexing.qdrant_manager import check_qdrant_health

        qdrant = check_qdrant_health()
        health["qdrant_connected"] = qdrant.get("connected", False)

        embedder = get_text_embedder()
        health["text_embedder_loaded"] = True
        health["text_embedder_dim"] = embedder.embedding_dim
    except Exception as e:
        health["qdrant_connected"] = False
        health["text_embedder_loaded"] = False

    health["ready"] = (
        health.get("ollama_reachable", False)
        and health.get("qdrant_connected", False)
        and health.get("text_embedder_loaded", False)
    )

    return health
