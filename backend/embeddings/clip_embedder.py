"""CLIP image embedder using open_clip.

Model      : ViT-B-32
Pretrained : laion2b_s34b_b79k
Output     : 512-dimensional L2-normalised image embedding
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image

from backend.config import settings
from backend.processors.schemas import ClipEmbeddingResult
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: "ClipEmbedder | None" = None


def get_clip_embedder() -> "ClipEmbedder":
    """Return the lazily-initialised singleton ClipEmbedder."""
    global _instance
    if _instance is None:
        _instance = ClipEmbedder()
    return _instance


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class ClipEmbedder:
    """Wraps open_clip ViT-B-32 for 512-d image embeddings."""

    def __init__(self) -> None:
        model_name = settings.clip_model_name
        pretrained = settings.clip_pretrained
        logger.info(
            "Loading CLIP model: %s (pretrained=%s) …", model_name, pretrained
        )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()

        self.model_label = f"{model_name}/{pretrained}"
        self._expected_dim = 512

        logger.info("ClipEmbedder ready — model=%s", self.model_label)

    # ---- public API ----

    def embed_image(self, image_path: str | Path) -> ClipEmbeddingResult:
        """Embed a local image and return a 512-d L2-normalised vector.

        Parameters
        ----------
        image_path : str or Path
            Path to a PNG / JPG / JPEG / WEBP image file.

        Raises
        ------
        FileNotFoundError
            If *image_path* does not exist.
        RuntimeError
            If the output dimension is not exactly 512.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            # L2-normalise
            features = features / features.norm(dim=-1, keepdim=True)

        vec = features.squeeze(0).cpu().numpy().astype(np.float32)
        dim = int(vec.shape[0])

        if dim != self._expected_dim:
            raise RuntimeError(
                f"ClipEmbedder produced {dim}-dim vector; expected {self._expected_dim}"
            )

        return ClipEmbeddingResult(
            vector=vec.tolist(),
            vector_dimension=dim,
            embedding_model=self.model_label,
            image_path=str(path),
            preprocessing="open_clip standard (resize + centre-crop + normalise)",
        )
