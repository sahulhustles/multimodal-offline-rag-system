"""Qdrant connection management and collection lifecycle.

Provides:
    get_qdrant_client()  — singleton Qdrant client
    ensure_collection()  — idempotent creation of the multimodal_rag collection
    get_collection_info() — collection metadata and vector configuration
    check_qdrant_health() — connectivity check
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
