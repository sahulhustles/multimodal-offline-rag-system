"""Vision processor — combines CLIP embedding + LLaVA description.

For every image this module produces:
- A 512-d CLIP embedding (image named vector).
- A LLaVA text description + its 384-d SentenceTransformer embedding
  (text named vector).

These will become **two linked Qdrant points** in Phase 3:
    A. Image/CLIP record  (image vector only)
    B. Description record (text vector only)

If CLIP succeeds but LLaVA fails, only the CLIP result is returned
with ``ingestion_status = "partial_failed"``.  No fake descriptions
are generated.
"""

from __future__ import annotations

from pathlib import Path

from backend.core.exceptions import ModelUnavailableError
from backend.embeddings.clip_embedder import get_clip_embedder
from backend.embeddings.text_embedder import get_text_embedder
from backend.processors.ollama_client import OllamaClient
from backend.processors.schemas import (
    ClipEmbeddingResult,
    ProcessorWarning,
    TextEmbeddingResult,
    VisionProcessingResult,
)
from backend.utils.file_utils import compute_file_hash
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton Ollama client
# ---------------------------------------------------------------------------

_ollama: OllamaClient | None = None


def _get_ollama() -> OllamaClient:
    global _ollama
    if _ollama is None:
        _ollama = OllamaClient()
    return _ollama


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_image(
    image_path: str | Path,
    source_document_id: str,
    original_filename: str,
    page_number: int | None = None,
) -> VisionProcessingResult:
    """Run the full vision pipeline on a single image.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file (PNG / JPG / JPEG / WEBP).
    source_document_id : str
        UUID of the parent source document.
    original_filename : str
        Original user-facing filename.
    page_number : int, optional
        PDF page number (1-based), if this image was extracted from a PDF.

    Returns
    -------
    VisionProcessingResult
        Contains CLIP embedding, LLaVA description + its embedding,
        status flags, and metadata.
    """
    image_path = Path(image_path)
    warnings: list[ProcessorWarning] = []

    # --- Image hash ---
    img_hash: str | None = None
    if image_path.exists():
        img_hash = compute_file_hash(image_path)

    # ------------------------------------------------------------------
    # Step 1: CLIP embedding (512-d)
    # ------------------------------------------------------------------
    clip_result: ClipEmbeddingResult | None = None
    clip_status = "failed"

    try:
        clip_embedder = get_clip_embedder()
        clip_result = clip_embedder.embed_image(image_path)
        clip_status = "completed"
        logger.info(
            "CLIP embedding completed: %s (%d-d)",
            image_path.name,
            clip_result.vector_dimension,
        )
    except Exception as exc:
        clip_status = "failed"
        warnings.append(
            ProcessorWarning(
                processor="vision_processor",
                message=f"CLIP embedding failed for {image_path.name}",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
        logger.error("CLIP embedding failed for %s: %s", image_path.name, exc)

    # ------------------------------------------------------------------
    # Step 2: LLaVA description via Ollama
    # ------------------------------------------------------------------
    llava_description: str | None = None
    llava_status = "failed"

    try:
        ollama = _get_ollama()
        llava_description = ollama.describe_image(image_path)
        llava_status = "completed"
        logger.info(
            "LLaVA description received: %d chars for %s",
            len(llava_description),
            image_path.name,
        )
    except ModelUnavailableError as exc:
        llava_status = "unavailable"
        warnings.append(
            ProcessorWarning(
                processor="vision_processor",
                message=f"LLaVA unavailable for {image_path.name}",
                detail=str(exc),
            )
        )
        logger.warning("LLaVA unavailable for %s: %s", image_path.name, exc)
    except Exception as exc:
        llava_status = "failed"
        warnings.append(
            ProcessorWarning(
                processor="vision_processor",
                message=f"LLaVA description failed for {image_path.name}",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
        logger.error("LLaVA failed for %s: %s", image_path.name, exc)

    # ------------------------------------------------------------------
    # Step 3: Embed the LLaVA description (384-d) — only if description exists
    # ------------------------------------------------------------------
    llava_desc_embedding: TextEmbeddingResult | None = None

    if llava_description:
        try:
            text_embedder = get_text_embedder()
            llava_desc_embedding = text_embedder.embed(llava_description)
            logger.info(
                "LLaVA description embedded: %d-d for %s",
                llava_desc_embedding.vector_dimension,
                image_path.name,
            )
        except Exception as exc:
            llava_desc_embedding = None
            warnings.append(
                ProcessorWarning(
                    processor="vision_processor",
                    message=f"Failed to embed LLaVA description for {image_path.name}",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            logger.error("Failed to embed LLaVA description for %s: %s", image_path.name, exc)

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------
    if clip_status == "completed" and llava_status == "completed":
        ingestion_status = "completed"
    elif clip_status == "completed":
        # CLIP worked, LLaVA didn't → partial failure
        ingestion_status = "partial_failed"
    else:
        # CLIP failed → full failure
        ingestion_status = "failed"

    return VisionProcessingResult(
        source_document_id=source_document_id,
        image_path=str(image_path),
        original_filename=original_filename,
        clip_embedding=clip_result,
        clip_status=clip_status,
        llava_description=llava_description,
        llava_description_embedding=llava_desc_embedding,
        llava_status=llava_status,
        ingestion_status=ingestion_status,
        page_number=page_number,
        image_hash=img_hash,
        warnings=warnings,
    )
