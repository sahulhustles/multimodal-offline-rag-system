"""Health check and collection statistics endpoints.

GET /api/v1/health  — Backend + Qdrant + Ollama connectivity, model config.
GET /api/v1/stats   — Qdrant collection point counts and breakdowns.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter

from backend.config import settings
from backend.indexing.qdrant_manager import (
    check_qdrant_health,
    get_collection_info,
    get_collection_stats_breakdown,
)
from backend.api.schemas.health import (
    HealthResponse,
    QdrantHealthStatus,
    OllamaHealthStatus,
    ModelsInfo,
    StatsResponse,
    CollectionInfo,
    StatsBreakdown,
)
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check backend, Qdrant, and Ollama connectivity and model availability."""

    # --- Qdrant ---
    qdrant_raw = check_qdrant_health()
    qdrant_status = QdrantHealthStatus(**qdrant_raw)

    # --- Ollama ---
    ollama_connected = False
    ollama_model_available = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                ollama_connected = True
                model_list = resp.json().get("models", [])
                ollama_model_available = any(
                    settings.ollama_model in m.get("name", "")
                    for m in model_list
                )
    except Exception as exc:
        logger.debug("Ollama health check failed: %s", exc)

    ollama_status = OllamaHealthStatus(
        connected=ollama_connected,
        model_available=ollama_model_available,
    )

    # --- Overall ---
    overall = "healthy" if qdrant_status.connected else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        mock_mode=settings.enable_demo_mock_mode,
        qdrant=qdrant_status,
        ollama=ollama_status,
        models=ModelsInfo(
            sentence_transformer=settings.sentence_transformer_model,
            clip=f"{settings.clip_model_name}/{settings.clip_pretrained}",
            whisper=f"{settings.whisper_model_size} ({settings.whisper_compute_type})",
        ),
    )


@router.get("/stats", response_model=StatsResponse)
async def collection_stats() -> StatsResponse:
    """Return Qdrant collection statistics and point-count breakdowns."""

    raw_info = get_collection_info()
    collection = CollectionInfo(**raw_info)

    breakdown_raw = get_collection_stats_breakdown()
    breakdown = StatsBreakdown(**breakdown_raw) if breakdown_raw else None

    return StatsResponse(collection=collection, breakdown=breakdown)
