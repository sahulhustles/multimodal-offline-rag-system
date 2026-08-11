"""Ollama local HTTP client for LLaVA image descriptions.

Calls the Ollama REST API at ``settings.ollama_base_url`` (default
``http://localhost:11434``) using the ``llava`` model.

API endpoint used::

    POST /api/generate
    {
      "model": "llava",
      "prompt": "<description prompt>",
      "images": ["<base64-encoded image>"],
      "stream": false
    }
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from backend.config import settings
from backend.core.exceptions import ModelUnavailableError
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# Default prompt for image description
DEFAULT_PROMPT = (
    "Describe this image in detail. Include all visible text, objects, "
    "charts, diagrams, colours, and spatial relationships. Be thorough "
    "and factual."
)


class OllamaClient:
    """Synchronous HTTP client for the local Ollama API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is pulled."""
        try:
            resp = httpx.get(
                f"{self.base_url}/api/tags", timeout=5.0
            )
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            return any(self.model in m.get("name", "") for m in models)
        except Exception:
            return False

    def describe_image(
        self,
        image_path: str | Path,
        prompt: str = DEFAULT_PROMPT,
    ) -> str:
        """Generate a text description of an image via LLaVA.

        Parameters
        ----------
        image_path : str or Path
            Path to the image file.
        prompt : str
            Prompt to guide the description.

        Returns
        -------
        str
            The LLaVA-generated description text.

        Raises
        ------
        ModelUnavailableError
            If Ollama is not reachable or the model is not available.
        FileNotFoundError
            If the image file does not exist.
        RuntimeError
            If the API returns an error response.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Encode image as base64
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Check availability first for a clearer error
        if not self.is_available():
            raise ModelUnavailableError(
                self.model,
                detail=(
                    f"Ollama is not reachable at {self.base_url} "
                    f"or model '{self.model}' is not pulled. "
                    f"Run: ollama pull {self.model}"
                ),
            )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }

        try:
            logger.info(
                "Requesting LLaVA description for %s (model=%s)",
                path.name,
                self.model,
            )
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama API returned status {resp.status_code}: "
                    f"{resp.text[:300]}"
                )

            data = resp.json()
            description = data.get("response", "").strip()

            if not description:
                raise RuntimeError(
                    "Ollama returned an empty response for image description."
                )

            logger.info(
                "LLaVA description received: %d chars for %s",
                len(description),
                path.name,
            )
            return description

        except ModelUnavailableError:
            raise
        except httpx.ConnectError:
            raise ModelUnavailableError(
                self.model,
                detail=f"Cannot connect to Ollama at {self.base_url}",
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s for {path.name}"
            )
