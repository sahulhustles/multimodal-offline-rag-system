"""Token-aware sliding-window text chunker.

Tokenizer
---------
WordPiece tokenizer from ``sentence-transformers/all-MiniLM-L6-v2``
(identical to ``bert-base-uncased`` vocabulary).  This is loaded via
``transformers.AutoTokenizer`` so that token counts are **exactly**
consistent with the downstream Sentence Transformer embedding model.

Configuration
-------------
- Chunk size : 512 tokens  (``settings.chunk_size_tokens``)
- Overlap    : 50 tokens   (``settings.chunk_overlap_tokens``)

Algorithm
---------
1. Tokenize the full input text once (source of truth for token IDs).
2. Identify sentence-boundary positions in the token stream as *hints*.
3. Walk a sliding window of ``chunk_size`` tokens:
   a. Prefer ending the window at the nearest sentence boundary
      in the **second half** of the window (avoids tiny chunks).
   b. If no suitable boundary exists, hard-cut at 512 tokens.
4. The next window starts ``overlap`` tokens before the previous end.
5. Each chunk records its ``page_number`` (for PDF-sourced text).
"""

from __future__ import annotations

import re
from typing import Optional

from transformers import AutoTokenizer

from backend.config import settings
from backend.processors.schemas import TextChunkResult
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton tokenizer — loaded once, reused everywhere
# ---------------------------------------------------------------------------

_tokenizer: AutoTokenizer | None = None


def get_tokenizer() -> AutoTokenizer:
    """Return the singleton WordPiece tokenizer for all-MiniLM-L6-v2."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            settings.sentence_transformer_model
        )
        logger.info(
            "Chunker tokenizer loaded: %s (vocab size %d)",
            settings.sentence_transformer_model,
            _tokenizer.vocab_size,
        )
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count the number of tokens in *text* using the model tokenizer."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Sentence splitting (heuristic)
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences with a lightweight regex heuristic.

    Splits on sentence-ending punctuation (``. ! ?``) followed by
    whitespace and an uppercase letter.  This avoids false splits on
    abbreviations like ``Dr.`` or ``U.S.`` in most cases.
    """
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _sentence_boundary_token_positions(
    text: str,
    tokenizer: AutoTokenizer,
    total_tokens: int,
) -> list[int]:
    """Return approximate token positions where sentences end.

    These positions are *hints* — the actual chunking uses the full-text
    token-ID array as the source of truth.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return []

    boundaries: list[int] = []
    cumulative = 0
    for sent in sentences[:-1]:          # skip last; it ends the text
        sent_toks = len(tokenizer.encode(sent, add_special_tokens=False))
        cumulative += sent_toks
        if 0 < cumulative < total_tokens:
            boundaries.append(cumulative)
    return boundaries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_chunks(
    text: str,
    source_document_id: str,
    page_number: Optional[int] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[TextChunkResult]:
    """Create sliding-window chunks from a single block of text.

    Parameters
    ----------
    text : str
        Input text.
    source_document_id : str
        UUID of the parent source document.
    page_number : int, optional
        PDF page number (1-based), if applicable.
    chunk_size : int, optional
        Maximum tokens per chunk.  Defaults to ``settings.chunk_size_tokens`` (512).
    overlap : int, optional
        Token overlap between consecutive chunks.  Defaults to
        ``settings.chunk_overlap_tokens`` (50).

    Returns
    -------
    list[TextChunkResult]
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size_tokens       # 512
    if overlap is None:
        overlap = settings.chunk_overlap_tokens       # 50

    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total = len(token_ids)

    if total == 0:
        return []

    # Fits in a single chunk
    if total <= chunk_size:
        return [
            TextChunkResult(
                source_document_id=source_document_id,
                chunk_index=0,
                text=text.strip(),
                token_count=total,
                page_number=page_number,
            )
        ]

    # Pre-compute sentence-boundary hints
    sentence_bounds = _sentence_boundary_token_positions(text, tokenizer, total)

    chunks: list[TextChunkResult] = []
    chunk_idx = 0
    start = 0

    while start < total:
        ideal_end = min(start + chunk_size, total)

        # Try to snap to a sentence boundary in the second half of the window
        end = ideal_end
        if ideal_end < total:
            min_end = start + chunk_size // 2
            best: int | None = None
            for b in sentence_bounds:
                if min_end < b <= ideal_end:
                    best = b
            if best is not None:
                end = best

        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

        chunks.append(
            TextChunkResult(
                source_document_id=source_document_id,
                chunk_index=chunk_idx,
                text=chunk_text,
                token_count=len(chunk_ids),
                page_number=page_number,
            )
        )
        chunk_idx += 1

        if end >= total:
            break

        # Next window starts ``overlap`` tokens before the current end
        start = end - overlap

    return chunks


def create_chunks_from_pages(
    pages: list[tuple[int, str]],
    source_document_id: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[TextChunkResult]:
    """Create chunks across multiple pages, preserving correct page numbers.

    Parameters
    ----------
    pages : list of (page_number, text) tuples
        Page-ordered text blocks from a PDF.
    source_document_id : str
        UUID of the parent source document.

    Returns
    -------
    list[TextChunkResult]
        Each chunk carries the ``page_number`` of the page where
        its first token originates.
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size_tokens
    if overlap is None:
        overlap = settings.chunk_overlap_tokens

    tokenizer = get_tokenizer()

    # Build a unified token stream with per-token page tracking
    all_token_ids: list[int] = []
    page_for_token: list[int] = []

    for page_num, text in pages:
        stripped = text.strip()
        if not stripped:
            continue
        page_tokens = tokenizer.encode(stripped, add_special_tokens=False)
        all_token_ids.extend(page_tokens)
        page_for_token.extend([page_num] * len(page_tokens))

    total = len(all_token_ids)
    if total == 0:
        return []

    if total <= chunk_size:
        full_text = tokenizer.decode(all_token_ids, skip_special_tokens=True).strip()
        return [
            TextChunkResult(
                source_document_id=source_document_id,
                chunk_index=0,
                text=full_text,
                token_count=total,
                page_number=page_for_token[0],
            )
        ]

    # Sentence boundaries across concatenated text
    full_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    sentence_bounds = _sentence_boundary_token_positions(
        full_text, tokenizer, total
    )

    chunks: list[TextChunkResult] = []
    chunk_idx = 0
    start = 0

    while start < total:
        ideal_end = min(start + chunk_size, total)
        end = ideal_end

        if ideal_end < total:
            min_end = start + chunk_size // 2
            best: int | None = None
            for b in sentence_bounds:
                if min_end < b <= ideal_end:
                    best = b
            if best is not None:
                end = best

        chunk_ids = all_token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

        chunks.append(
            TextChunkResult(
                source_document_id=source_document_id,
                chunk_index=chunk_idx,
                text=chunk_text,
                token_count=len(chunk_ids),
                page_number=page_for_token[start],
            )
        )
        chunk_idx += 1

        if end >= total:
            break

        start = end - overlap

    return chunks
