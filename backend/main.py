"""FastAPI application entrypoint for the Multimodal RAG System.

Startup sequence:
    1. Configure structured logging.
    2. Initialise SQLite database tables.
    3. Ensure Qdrant collection exists with named vectors.

Run with:
    python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.db.database import init_db
from backend.utils.logging_config import setup_logging, get_logger
from backend.api.router import api_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""

    # --- Startup ---
    setup_logging()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    if settings.enable_demo_mock_mode:
        logger.warning(
            "⚠️  DEMO MOCK MODE is ENABLED — no real model inference will occur"
        )

    # Initialise SQLite database
    init_db()

    # Initialise Qdrant collection
    try:
        from backend.indexing.qdrant_manager import ensure_collection

        ensure_collection()
        logger.info("Qdrant collection '%s' ready", settings.qdrant_collection_name)
    except Exception as e:
        logger.error("Failed to initialise Qdrant collection: %s", e)
        logger.warning(
            "Backend will start but vector operations will fail until Qdrant is available"
        )

    yield

    # --- Shutdown ---
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Fully Offline Multimodal Retrieval-Augmented Generation System. "
        "Ingests PDFs, DOC/DOCX, images, audio, and text notes — "
        "processes, embeds, and indexes into a local Qdrant vector database."
    ),
    lifespan=lifespan,
)

# CORS — allow frontend dev server and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all API routes under /api/v1
app.include_router(api_router, prefix="/api/v1")

# Mount demo files for UI preview
import os
from fastapi.staticfiles import StaticFiles
os.makedirs("data/demo", exist_ok=True)
app.mount("/demo-assets", StaticFiles(directory="data/demo"), name="demo_assets")
