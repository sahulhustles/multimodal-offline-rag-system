"""Ollama chat client for Phase 5 answer generation.

Uses the ``/api/chat`` endpoint (chat/instruct mode with system + user
messages).  This is separate from the existing ``OllamaClient`` in
``processors/ollama_client.py`` which uses ``/api/generate`` for LLaVA
image descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from backend.config import settings
from backend.core.exceptions import ModelUnavailableError
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ChatResponse:
    """Parsed response from Ollama /api/chat."""

    content: str
    model: str
    done: bool
    total_duration_ns: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0


class OllamaChatClient:
    """Synchronous HTTP client for Ollama /api/chat (text generation)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.generation_model
        self.timeout = timeout or settings.generation_timeout

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the generation model is pulled."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            return any(self.model in m.get("name", "") for m in models)
        except Exception:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to Ollama.

        Parameters
        ----------
        messages:
            List of ``{role, content}`` message dicts.
        temperature:
            Sampling temperature. Defaults to ``settings.generation_temperature``.
        max_tokens:
            Max output tokens. Defaults to ``settings.generation_max_tokens``.

        Returns
        -------
        ChatResponse
            Parsed response with content and timing metadata.

        Raises
        ------
        ModelUnavailableError
            If Ollama is unreachable or the model is not available.
        RuntimeError
            If the API returns an error or empty response.
        """
        effective_temp = temperature if temperature is not None else settings.generation_temperature
        effective_max = max_tokens if max_tokens is not None else settings.generation_max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": effective_temp,
                "num_predict": effective_max,
            },
        }

        try:
            logger.info(
                "Ollama chat request: model=%s, messages=%d, temp=%.1f, max_tokens=%d",
                self.model,
                len(messages),
                effective_temp,
                effective_max,
            )

            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama /api/chat returned status {resp.status_code}: "
                    f"{resp.text[:300]}"
                )

            data = resp.json()
            message = data.get("message", {})
            content = message.get("content", "").strip()

            if not content:
                raise RuntimeError(
                    "Ollama returned an empty response for chat completion."
                )

            response = ChatResponse(
                content=content,
                model=data.get("model", self.model),
                done=data.get("done", True),
                total_duration_ns=data.get("total_duration", 0),
                prompt_eval_count=data.get("prompt_eval_count", 0),
                eval_count=data.get("eval_count", 0),
            )

            logger.info(
                "Ollama chat response: %d chars, eval_count=%d, total_duration=%dms",
                len(content),
                response.eval_count,
                response.total_duration_ns // 1_000_000,
            )

            return response

        except ModelUnavailableError:
            raise
        except httpx.ConnectError:
            raise ModelUnavailableError(
                self.model,
                detail=f"Cannot connect to Ollama at {self.base_url}",
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama chat request timed out after {self.timeout}s "
                f"(model={self.model})"
            )


# Module-level singleton
_chat_client: OllamaChatClient | None = None


def get_chat_client() -> OllamaChatClient:
    """Get or create the module-level OllamaChatClient singleton."""
    global _chat_client
    if _chat_client is None:
        _chat_client = OllamaChatClient()
    return _chat_client
