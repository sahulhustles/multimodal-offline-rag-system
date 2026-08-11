"""Sentence Transformer text embedder.

Model : sentence-transformers/all-MiniLM-L6-v2
Output: 384-dimensional L2-normalised vector

Span-splitting strategy
-----------------------
all-MiniLM-L6-v2 has ``max_seq_length = 256`` tokens (including [CLS]
and [SEP]), giving 254 usable content tokens.  The chunker produces
512-token chunks, so every chunk is split into ≤254-token spans:

1. Tokenize the chunk with the model's own tokenizer.
2. Split into spans of at most ``max_seq_length - 2`` tokens.
3. Encode each span independently.
4. **Mean-pool** the span embeddings.
5. **L2-normalise** the final vector.

The full 512-token source text is always preserved in the result.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.processors.schemas import TextEmbeddingResult
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: "TextEmbedder | None" = None


def get_text_embedder() -> "TextEmbedder":
    """Return the lazily-initialised singleton TextEmbedder."""
    global _instance
    if _instance is None:
        _instance = TextEmbedder()
    return _instance


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class TextEmbedder:
    """Wraps SentenceTransformer with span-split + mean-pool for long chunks."""

    def __init__(self) -> None:
        model_name = settings.sentence_transformer_model
        logger.info("Loading SentenceTransformer model: %s …", model_name)

        self.model = SentenceTransformer(model_name)
        self.model_name: str = model_name

        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        if self.embedding_dim != 384:
            raise RuntimeError(
                f"Expected 384-dim embeddings from {model_name}, "
                f"got {self.embedding_dim}"
            )

        self.max_seq_length: int = self.model.max_seq_length  # 256
        # Usable content tokens = max_seq_length - 2 ([CLS] + [SEP])
        self._usable_tokens: int = self.max_seq_length - 2

        logger.info(
            "TextEmbedder ready — dim=%d  max_seq_length=%d  usable=%d",
            self.embedding_dim,
            self.max_seq_length,
            self._usable_tokens,
        )

    # ---- public API ----

    def embed(self, text: str) -> TextEmbeddingResult:
        """Embed *text* and return a 384-d L2-normalised vector.

        Raises ``RuntimeError`` if the output dimension is not 384.
        """
        tokenizer = self.model.tokenizer
        token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(token_ids)

        if token_count <= self._usable_tokens:
            # Fits in a single forward pass
            vec = self.model.encode(text, normalize_embeddings=True)
            return self._build_result(vec, span_count=1, text=text, tokens=token_count)

        # Span-split → encode → mean-pool → L2-normalise
        spans = self._split_spans(token_ids, tokenizer)
        span_texts = [
            tokenizer.decode(s, skip_special_tokens=True) for s in spans
        ]

        raw_vecs = self.model.encode(span_texts, normalize_embeddings=False)
        mean_vec = np.mean(raw_vecs, axis=0)

        # L2-normalise
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm

        return self._build_result(
            mean_vec, span_count=len(spans), text=text, tokens=token_count
        )

    def embed_batch(self, texts: list[str]) -> list[TextEmbeddingResult]:
        """Embed a list of texts. Convenience wrapper around :meth:`embed`."""
        return [self.embed(t) for t in texts]

    # ---- internals ----

    def _split_spans(
        self, token_ids: list[int], tokenizer
    ) -> list[list[int]]:
        """Split *token_ids* into spans of ≤ ``_usable_tokens`` length."""
        spans: list[list[int]] = []
        pos = 0
        while pos < len(token_ids):
            end = min(pos + self._usable_tokens, len(token_ids))
            spans.append(token_ids[pos:end])
            pos = end
        return spans

    def _build_result(
        self,
        vec: np.ndarray,
        span_count: int,
        text: str,
        tokens: int,
    ) -> TextEmbeddingResult:
        dim = int(vec.shape[0])
        if dim != 384:
            raise RuntimeError(
                f"TextEmbedder produced {dim}-dim vector; expected 384"
            )
        return TextEmbeddingResult(
            vector=vec.tolist(),
            vector_dimension=dim,
            embedding_model=self.model_name,
            embedding_span_count=span_count,
            source_text=text,
            token_count=tokens,
        )
