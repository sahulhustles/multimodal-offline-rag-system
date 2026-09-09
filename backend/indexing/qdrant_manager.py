"""Qdrant connection management, collection lifecycle, and indexing utilities.

Provides:
    get_qdrant_client()       — singleton Qdrant client
    ensure_collection()       — idempotent creation of the multimodal_rag collection
    ensure_payload_indexes()  — create payload indexes for efficient filtering
    get_collection_info()     — collection metadata and vector configuration
    check_qdrant_health()     — connectivity check
    upsert_points_batch()     — batch upsert with configurable size
    delete_points_by_source() — filtered delete by source_document_id
    count_points_by_source()  — count points for a source document
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from backend.config import settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# Singleton client
_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return the singleton Qdrant client, creating it on first call."""
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        logger.info(
            "Qdrant client connected to %s:%s",
            settings.qdrant_host,
            settings.qdrant_port,
        )
    return _client


def ensure_collection() -> None:
    """Idempotently create the multimodal_rag collection with named vectors.

    Named vectors:
        text  — 384 dimensions, cosine distance (SentenceTransformer embeddings)
        image — 512 dimensions, cosine distance (CLIP embeddings)

    If the collection already exists, this is a no-op.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    # Check if collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.info("Qdrant collection '%s' already exists — skipping creation", collection_name)
        return

    # Create with named vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
            "image": models.VectorParams(
                size=512,
                distance=models.Distance.COSINE,
            ),
        },
    )
    logger.info(
        "Created Qdrant collection '%s' with named vectors: "
        "text (384d, cosine), image (512d, cosine)",
        collection_name,
    )

    # Create payload indexes for efficient filtering (Phase 3)
    ensure_payload_indexes()


def get_collection_info() -> dict:
    """Return collection metadata including vector config and point counts.

    Returns a dict with keys: collection_name, points_count, vectors_count,
    indexed_vectors_count, status, vectors_config.  On failure, includes
    an 'error' key instead.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    try:
        info = client.get_collection(collection_name)

        # Extract vector configurations
        vectors_config: dict = {}
        raw_vectors = info.config.params.vectors
        if isinstance(raw_vectors, dict):
            for name, params in raw_vectors.items():
                vectors_config[name] = {
                    "size": params.size,
                    "distance": params.distance.value if params.distance else "unknown",
                }

        return {
            "collection_name": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value if info.status else "unknown",
            "vectors_config": vectors_config,
        }
    except Exception as e:
        logger.error("Failed to get collection info: %s", e)
        return {"collection_name": collection_name, "error": str(e)}


def get_collection_stats_breakdown() -> dict | None:
    """Return point-count breakdowns by source_type, modality, and ingestion_status.

    Returns None if stats cannot be computed (e.g. collection doesn't exist).
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    try:
        breakdown: dict = {
            "by_source_type": {},
            "by_modality": {},
            "by_ingestion_status": {},
        }

        # Count by source_type
        source_types = [
            "pdf", "doc", "docx", "image", "screenshot",
            "pdf_extracted_image", "image_description",
            "screenshot_description", "pdf_image_description",
            "audio", "text_note",
        ]
        for st in source_types:
            count_result = client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_type",
                            match=models.MatchValue(value=st),
                        )
                    ]
                ),
            )
            if count_result.count > 0:
                breakdown["by_source_type"][st] = count_result.count

        # Count by modality
        for mod in ["text", "image", "audio"]:
            count_result = client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="modality",
                            match=models.MatchValue(value=mod),
                        )
                    ]
                ),
            )
            if count_result.count > 0:
                breakdown["by_modality"][mod] = count_result.count

        # Count by ingestion_status
        for status in ["completed", "partial_failed", "failed"]:
            count_result = client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="ingestion_status",
                            match=models.MatchValue(value=status),
                        )
                    ]
                ),
            )
            if count_result.count > 0:
                breakdown["by_ingestion_status"][status] = count_result.count

        return breakdown
    except Exception as e:
        logger.warning("Could not compute collection stats breakdown: %s", e)
        return None


def check_qdrant_health() -> dict:
    """Check Qdrant connectivity by listing collections.

    Returns {"connected": True/False, ...}.
    """
    try:
        client = get_qdrant_client()
        client.get_collections()
        return {
            "connected": True,
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Phase 3: Payload indexes
# ---------------------------------------------------------------------------

_PAYLOAD_INDEX_FIELDS = [
    ("source_document_id", models.PayloadSchemaType.KEYWORD),
    ("modality", models.PayloadSchemaType.KEYWORD),
    ("source_type", models.PayloadSchemaType.KEYWORD),
    ("indexing_status", models.PayloadSchemaType.KEYWORD),
]


def ensure_payload_indexes() -> None:
    """Create payload indexes on the collection for efficient filtering.

    Idempotent — Qdrant silently ignores indexes that already exist.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    for field_name, schema_type in _PAYLOAD_INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
        except Exception as exc:
            # Some Qdrant versions raise if index exists — swallow it
            logger.debug(
                "Payload index '%s' may already exist: %s", field_name, exc
            )

    logger.info(
        "Payload indexes ensured for %s: %s",
        collection_name,
        [f for f, _ in _PAYLOAD_INDEX_FIELDS],
    )


# ---------------------------------------------------------------------------
# Phase 3: Batch upsert
# ---------------------------------------------------------------------------


def upsert_points_batch(points: list[models.PointStruct]) -> int:
    """Upsert points into the collection in configurable batches.

    Returns the total number of points upserted.
    """
    if not points:
        return 0

    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    batch_size = settings.qdrant_upsert_batch_size
    total = 0

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )
        total += len(batch)
        logger.debug(
            "Upserted batch %d–%d (%d points)", i, i + len(batch) - 1, len(batch)
        )

    logger.info("Upserted %d points into '%s'", total, collection_name)
    return total


# ---------------------------------------------------------------------------
# Phase 3: Filtered delete
# ---------------------------------------------------------------------------


def delete_points_by_source(
    source_document_id: str, *, wait: bool = True
) -> None:
    """Delete all Qdrant points belonging to a source document.

    Uses a payload filter on ``source_document_id``.  When *wait* is
    True, the call blocks until deletion is confirmed.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_document_id",
                        match=models.MatchValue(value=source_document_id),
                    )
                ]
            )
        ),
        wait=wait,
    )
    logger.info(
        "Deleted Qdrant points for source_document_id=%s (wait=%s)",
        source_document_id,
        wait,
    )


# ---------------------------------------------------------------------------
# Phase 3: Count by source
# ---------------------------------------------------------------------------


def count_points_by_source(source_document_id: str) -> int:
    """Return the number of Qdrant points for a given source document."""
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    result = client.count(
        collection_name=collection_name,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_document_id",
                    match=models.MatchValue(value=source_document_id),
                )
            ]
        ),
    )
    return result.count
