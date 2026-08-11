"""Main API router — aggregates all endpoint routers under /api/v1."""

from fastapi import APIRouter

from backend.api.endpoints import health

api_router = APIRouter()

# Health & Stats
api_router.include_router(health.router, tags=["Health & Stats"])

# Future routers will be added here in subsequent phases:
# api_router.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
# api_router.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
# api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
# api_router.include_router(records.router, prefix="/records", tags=["Records"])

from backend.api.endpoints import demo
api_router.include_router(demo.router, prefix="/demo", tags=["Demo Processing APIs"])
