"""Generation orchestrator with interactive conversation and grounded answering.

Pipeline:
    1. Classify query intent (CONVERSATIONAL vs DOCUMENT_QUERY).
    2. If CONVERSATIONAL:
       - Build interactive chat prompt with history and multimodal assistant persona.
       - Generate response via Ollama /api/chat.
       - Return AnswerResponse with is_conversational=True, grounded=False.
    3. If DOCUMENT_QUERY:
       - Contextualize query with recent history for pronoun/reference resolution.
       - Call existing Phase 4 retrieve() with contextualized query.
       - Evidence gate — return insufficient-context response if no usable results.
       - Assemble context with assemble_context().
       - Build grounded prompt with build_messages() (including history turns).
       - Call Ollama via OllamaChatClient.chat().
       - Validate citations with validate_citations().
       - Return AnswerResponse with citations and provenance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from backend.config import settings
from backend.generation.citation_validator import (
    VALIDATION_FAILURE_MESSAGE,
    CitationResult,
    validate_citations,
)
from backend.generation.context_assembler import AssembledContext, SourceBlock, assemble_context
from backend.generation.intent_router import QueryIntent, classify_intent, contextualize_query
from backend.generation.ollama_chat import ChatResponse, get_chat_client
from backend.generation.prompt_builder import (
    INSUFFICIENT_CONTEXT_PHRASE,
    build_conversational_messages,
    build_messages,
)
from backend.retrieval.retrieval import RetrievalResult, SearchResponse, retrieve
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CitationInfo:
    """Citation metadata for a single source in the answer."""

    source_id: str
    chunk_id: str
    source_document_id: str
    filename: str | None
    source_type: str
    modality: str
    page_number: int | None
    chunk_index: int
    timestamp_start: float | None
    timestamp_end: float | None
    score: float
    preview: str


@dataclass
class RetrievalMetrics:
    """Phase 4 retrieval timing metrics."""

    encoding_time_ms: float
    search_time_ms: float
    hydration_time_ms: float
    total_time_ms: float
    result_count: int
    embedding_model: str
    query_vector_dim: int


@dataclass
class GenerationMetrics:
    """Phase 5 generation timing and stats."""

    model: str
    generation_time_ms: float
    prompt_eval_count: int
    eval_count: int
    context_record_count: int
    context_char_count: int


@dataclass
class AnswerResponse:
    """Complete answer response supporting conversational and grounded modes."""

    query: str
    answer: str
    grounded: bool
    insufficient_context: bool
    is_conversational: bool = False
    citations: list[CitationInfo] = field(default_factory=list)
    retrieval_metrics: RetrievalMetrics | None = None
    generation_metrics: GenerationMetrics | None = None
    warnings: list[str] = field(default_factory=list)
    retrieval_results: list[RetrievalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_answer(
    query: str,
    source_document_ids: list[str],
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    conversation_history: Sequence[dict[str, str]] | None = None,
) -> AnswerResponse:
    """Execute the conversational / grounded generation pipeline.

    Parameters
    ----------
    query:
        Natural-language user message or question.
    source_document_ids:
        Source document IDs to search within (already validated as indexed).
    top_k:
        Max retrieval results. Defaults to ``settings.retrieval_top_k``.
    score_threshold:
        Retrieval score threshold. Defaults to ``settings.retrieval_score_threshold``.
    conversation_history:
        Recent conversation turns for multi-turn dialogue context.

    Returns
    -------
    AnswerResponse
        Grounded answer with citations, conversational response, or error state.
    """
    # ---- Step 1: Detect intent ----
    intent = classify_intent(query)
    logger.info("Query '%s' classified as %s", query[:50], intent.value)

    # ---- Conversational path ----
    if intent == QueryIntent.CONVERSATIONAL:
        messages = build_conversational_messages(query, conversation_history)
        t_gen_start = time.perf_counter()
        try:
            client = get_chat_client()
            chat_response: ChatResponse = client.chat(messages)
        except Exception as e:
            logger.error("Conversational generation failed: %s", e, exc_info=True)
            return AnswerResponse(
                query=query,
                answer="Hello! I'm here to help you search and analyze your documents. How can I assist you today?",
                grounded=False,
                insufficient_context=False,
                is_conversational=True,
                warnings=[f"Conversational generation fallback: {str(e)}"],
            )
        t_gen_end = time.perf_counter()

        generation_metrics = GenerationMetrics(
            model=chat_response.model,
            generation_time_ms=round((t_gen_end - t_gen_start) * 1000, 1),
            prompt_eval_count=chat_response.prompt_eval_count,
            eval_count=chat_response.eval_count,
            context_record_count=0,
            context_char_count=0,
        )

        return AnswerResponse(
            query=query,
            answer=chat_response.content,
            grounded=False,
            insufficient_context=False,
            is_conversational=True,
            generation_metrics=generation_metrics,
        )

    # ---- Document Query path ----
    # Contextualize query with recent turns if follow-up pronouns exist
    retrieval_query = contextualize_query(query, conversation_history)

    # Step 2: Phase 4 retrieval
    search_response: SearchResponse = retrieve(
        query=retrieval_query,
        source_document_ids=source_document_ids,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    retrieval_metrics = RetrievalMetrics(
        encoding_time_ms=search_response.encoding_time_ms,
        search_time_ms=search_response.search_time_ms,
        hydration_time_ms=search_response.hydration_time_ms,
        total_time_ms=search_response.total_time_ms,
        result_count=search_response.result_count,
        embedding_model=search_response.embedding_model,
        query_vector_dim=search_response.query_vector_dim,
    )

    # Step 3: Evidence gate
    if not search_response.results:
        logger.info("Evidence gate: no retrieval results — returning insufficient context")
        return AnswerResponse(
            query=query,
            answer=INSUFFICIENT_CONTEXT_PHRASE,
            grounded=False,
            insufficient_context=True,
            is_conversational=False,
            retrieval_metrics=retrieval_metrics,
            warnings=["No retrieval results found for the query."],
            retrieval_results=search_response.results,
        )

    # Step 4: Assemble context
    assembled: AssembledContext = assemble_context(search_response.results)

    if not assembled.blocks:
        logger.info(
            "Evidence gate: all %d results dropped (low_score=%d, dedup=%d) — insufficient context",
            len(search_response.results),
            assembled.dropped_low_score,
            assembled.dropped_dedup,
        )
        return AnswerResponse(
            query=query,
            answer=INSUFFICIENT_CONTEXT_PHRASE,
            grounded=False,
            insufficient_context=True,
            is_conversational=False,
            retrieval_metrics=retrieval_metrics,
            warnings=[
                f"No retrieved evidence passed the minimum score threshold "
                f"({settings.generation_min_score})."
            ],
            retrieval_results=search_response.results,
        )

    # Step 5: Build grounded prompt (including conversation history)
    messages = build_messages(query, assembled, conversation_history)

    # Step 6: Call Ollama
    t_gen_start = time.perf_counter()
    try:
        client = get_chat_client()
        chat_response: ChatResponse = client.chat(messages)
    except Exception as e:
        logger.error("Ollama generation failed: %s", e, exc_info=True)
        return AnswerResponse(
            query=query,
            answer=VALIDATION_FAILURE_MESSAGE,
            grounded=False,
            insufficient_context=False,
            is_conversational=False,
            retrieval_metrics=retrieval_metrics,
            warnings=[f"Generation failed: {str(e)}"],
            retrieval_results=search_response.results,
        )
    t_gen_end = time.perf_counter()
    generation_time_ms = (t_gen_end - t_gen_start) * 1000

    generation_metrics = GenerationMetrics(
        model=chat_response.model,
        generation_time_ms=round(generation_time_ms, 1),
        prompt_eval_count=chat_response.prompt_eval_count,
        eval_count=chat_response.eval_count,
        context_record_count=len(assembled.blocks),
        context_char_count=assembled.total_chars,
    )

    # Step 7: Validate citations
    citation_result: CitationResult = validate_citations(
        chat_response.content,
        assembled.source_id_set,
    )

    # Build citation info for valid cited sources
    citations: list[CitationInfo] = []
    block_map = assembled.source_id_to_block
    for sid in citation_result.cited_ids:
        block = block_map.get(sid)
        if block:
            citations.append(
                CitationInfo(
                    source_id=sid,
                    chunk_id=block.chunk_id,
                    source_document_id=block.source_document_id,
                    filename=block.filename,
                    source_type=block.source_type,
                    modality=block.modality,
                    page_number=block.page_number,
                    chunk_index=block.chunk_index,
                    timestamp_start=block.timestamp_start,
                    timestamp_end=block.timestamp_end,
                    score=block.score,
                    preview=block.preview,
                )
            )

    return AnswerResponse(
        query=query,
        answer=citation_result.answer,
        grounded=citation_result.grounded,
        insufficient_context=citation_result.insufficient_context,
        is_conversational=False,
        citations=citations,
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        warnings=citation_result.warnings,
        retrieval_results=search_response.results,
    )
