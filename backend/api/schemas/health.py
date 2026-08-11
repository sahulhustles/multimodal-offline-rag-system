"""Pydantic response models for health and stats endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class QdrantHealthStatus(BaseModel):
    """Qdrant connectivity status."""

    connected: bool
    host: Optional[str] = None
    port: Optional[int] = None
    error: Optional[str] = None


class OllamaHealthStatus(BaseModel):
    """Ollama connectivity and model availability status."""

    connected: bool
    model_available: bool


class ModelsInfo(BaseModel):
    """Configured ML model identifiers."""

    sentence_transformer: str
    clip: str
    whisper: str


class HealthResponse(BaseModel):
    """Response for GET /api/v1/health."""

    status: str  # "healthy" | "degraded"
    version: str
    mock_mode: bool
    qdrant: QdrantHealthStatus
    ollama: OllamaHealthStatus
    models: ModelsInfo


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------


class VectorConfigInfo(BaseModel):
    """Vector configuration for a single named vector."""

    size: int
    distance: str


class CollectionInfo(BaseModel):
    """Qdrant collection metadata."""

    collection_name: str
    points_count: Optional[int] = None
    vectors_count: Optional[int] = None
    indexed_vectors_count: Optional[int] = None
    status: Optional[str] = None
    vectors_config: Optional[dict[str, VectorConfigInfo]] = None
    error: Optional[str] = None


class StatsBreakdown(BaseModel):
    """Point-count breakdowns by various dimensions."""

    by_source_type: dict[str, int] = {}
    by_modality: dict[str, int] = {}
    by_ingestion_status: dict[str, int] = {}


class StatsResponse(BaseModel):
    """Response for GET /api/v1/stats."""

    collection: CollectionInfo
    breakdown: Optional[StatsBreakdown] = None
