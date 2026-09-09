"""Low-level Qdrant text-vector search wrapper.

Provides a thin abstraction over ``qdrant_client.query_points`` for the
``text`` named-vector space.  All filter construction happens here so the
retrieval engine only needs to pass high-level parameters.
"""

from __future__ import annotations

from qdrant_client import models

from backend.config import settings
from backend.indexing.qdrant_manager import get_qdrant_client
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


def search_text_vectors(
    query_vector: list[float],
    source_document_ids: list[str],
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    modality_filter: str | None = None,
) -> list[models.ScoredPoint]:
    """Search the ``text`` named-vector space in Qdrant.

    Parameters
    ----------
    query_vector:
        384-dimensional query vector from SentenceTransformer.
    source_document_ids:
        Only search points belonging to these source documents.
    limit:
        Max number of results to return.
    score_threshold:
        Minimum cosine similarity score.  ``0.0`` disables thresholding.
    modality_filter:
        Optional modality filter (``text``, ``image``, ``audio``).

    Returns
    -------
    list[models.ScoredPoint]
        Scored points with payloads, ordered by descending score.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    # --- Build filter conditions ---
    must_conditions: list[models.Condition] = [
        # Source document scope
        models.FieldCondition(
            key="source_document_id",
            match=models.MatchAny(any=source_document_ids),
        ),
        # Only search indexed points
        models.FieldCondition(
            key="indexing_status",
            match=models.MatchAny(any=["completed", "indexed"]),
        ),
    ]

    if modality_filter:
        must_conditions.append(
            models.FieldCondition(
                key="modality",
                match=models.MatchValue(value=modality_filter),
            )
        )

    query_filter = models.Filter(must=must_conditions)

    # --- Execute search ---
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using="text",
        query_filter=query_filter,
        limit=limit,
        score_threshold=score_threshold if score_threshold > 0 else None,
        with_payload=True,
    )

    logger.info(
        "Qdrant text-vector search: %d results (limit=%d, docs=%d, threshold=%.3f)",
        len(results.points),
        limit,
        len(source_document_ids),
        score_threshold,
    )

    return results.points
