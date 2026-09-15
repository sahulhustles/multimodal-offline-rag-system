"""Context assembler for Phase 5 generation.

Consumes Phase 4 ``RetrievalResult`` objects, deduplicates overlapping
content, enforces configurable context-budget limits, and builds numbered
``[S#]`` source blocks with metadata headers suitable for an LLM prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.config import settings
from backend.retrieval.retrieval import RetrievalResult
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SourceBlock:
    """A single assembled source block for the LLM context."""

    source_id: str  # "S1", "S2", ...
    chunk_id: str
    source_document_id: str
    filename: str | None
    source_type: str
    modality: str
    page_number: int | None
    chunk_index: int
    timestamp_start: float | None
    timestamp_end: float | None
    score: float
    content: str
    preview: str  # First ~200 chars for the citation response


@dataclass
class AssembledContext:
    """The full assembled context ready for prompt building."""

    blocks: list[SourceBlock] = field(default_factory=list)
    total_chars: int = 0
    dropped_low_score: int = 0
    dropped_dedup: int = 0
    dropped_budget: int = 0

    @property
    def source_id_set(self) -> set[str]:
        """Set of allowed source IDs (e.g. {'S1', 'S2', 'S3'})."""
        return {b.source_id for b in self.blocks}

    @property
    def source_id_to_block(self) -> dict[str, SourceBlock]:
        """Map source_id → SourceBlock for citation resolution."""
        return {b.source_id: b for b in self.blocks}

    def render(self) -> str:
        """Render all source blocks as a single string for the LLM."""
        parts: list[str] = []
        for block in self.blocks:
            header = _build_header(block)
            parts.append(f"{header}\n---\n{block.content}")
        return "\n\n".join(parts)


def _build_header(block: SourceBlock) -> str:
    """Build the [S#] header line for a source block."""
    parts = [f"[{block.source_id}]"]

    if block.filename:
        parts.append(block.filename)

    parts.append(block.modality)

    if block.page_number is not None:
        parts.append(f"page {block.page_number}")

    parts.append(f"chunk #{block.chunk_index}")

    if block.timestamp_start is not None:
        ts = f"{block.timestamp_start:.1f}s"
        if block.timestamp_end is not None:
            ts += f"–{block.timestamp_end:.1f}s"
        parts.append(ts)

    parts.append(f"score: {block.score:.2f}")

    return " | ".join(parts)


def _extract_content(result: RetrievalResult) -> str:
    """Extract the best available text content from a retrieval result."""
    if result.extracted_text:
        return result.extracted_text
    if result.transcript_text:
        return result.transcript_text
    if result.llava_description:
        return result.llava_description
    if result.text_preview:
        return result.text_preview
    return ""


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the nearest sentence boundary within max_chars."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Find last sentence-ending punctuation
    last_period = max(
        truncated.rfind(". "),
        truncated.rfind(".\n"),
        truncated.rfind("? "),
        truncated.rfind("! "),
    )

    if last_period > max_chars * 0.5:  # Only use if we keep >50% of content
        return truncated[: last_period + 1].rstrip()

    return truncated.rstrip() + "…"


def _text_overlap_ratio(a: str, b: str) -> float:
    """Compute approximate overlap ratio between two texts using character sets.

    Uses a cheap set-of-words approach rather than expensive diff.
    """
    if not a or not b:
        return 0.0

    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))

    if not words_a or not words_b:
        return 0.0

    intersection = len(words_a & words_b)
    smaller = min(len(words_a), len(words_b))

    return intersection / smaller if smaller > 0 else 0.0


def assemble_context(
    results: list[RetrievalResult],
    *,
    min_score: float | None = None,
    max_records: int | None = None,
    max_chars_per_record: int | None = None,
    max_total_chars: int | None = None,
) -> AssembledContext:
    """Assemble retrieved evidence into numbered source blocks for LLM context.

    Parameters
    ----------
    results:
        Phase 4 retrieval results, already in score-descending order.
    min_score:
        Drop results below this score. Defaults to ``settings.generation_min_score``.
    max_records:
        Maximum source blocks. Defaults to ``settings.generation_max_context_records``.
    max_chars_per_record:
        Per-record char limit. Defaults to ``settings.generation_max_chars_per_record``.
    max_total_chars:
        Total context char budget. Defaults to ``settings.generation_max_context_chars``.

    Returns
    -------
    AssembledContext
        Ready-to-render context with source blocks and budget stats.
    """
    effective_min_score = min_score if min_score is not None else settings.generation_min_score
    effective_max_records = max_records if max_records is not None else settings.generation_max_context_records
    effective_max_per = max_chars_per_record if max_chars_per_record is not None else settings.generation_max_chars_per_record
    effective_max_total = max_total_chars if max_total_chars is not None else settings.generation_max_context_chars

    ctx = AssembledContext()

    # Step 1: Filter by score
    scored_results: list[RetrievalResult] = []
    for r in results:
        if r.score >= effective_min_score:
            scored_results.append(r)
        else:
            ctx.dropped_low_score += 1

    # Step 2: Deduplicate overlapping content (keep higher-scored)
    deduped: list[RetrievalResult] = []
    accepted_texts: list[str] = []

    for r in scored_results:
        content = _extract_content(r)
        if not content.strip():
            ctx.dropped_dedup += 1
            continue

        is_duplicate = False
        for existing_text in accepted_texts:
            if _text_overlap_ratio(content, existing_text) > 0.80:
                is_duplicate = True
                break

        if is_duplicate:
            ctx.dropped_dedup += 1
        else:
            deduped.append(r)
            accepted_texts.append(content)

    # Step 3: Build blocks with per-record truncation and total budget
    source_num = 0
    for r in deduped:
        if source_num >= effective_max_records:
            ctx.dropped_budget += len(deduped) - source_num
            break

        content = _extract_content(r)
        content = _truncate_at_sentence(content, effective_max_per)

        # Check total budget
        if ctx.total_chars + len(content) > effective_max_total:
            # Try to fit a shorter version
            remaining = effective_max_total - ctx.total_chars
            if remaining > 200:  # Only include if we can fit >200 chars
                content = _truncate_at_sentence(content, remaining)
            else:
                ctx.dropped_budget += len(deduped) - source_num
                break

        source_num += 1
        source_id = f"S{source_num}"
        preview = content[:200].rstrip()

        block = SourceBlock(
            source_id=source_id,
            chunk_id=r.chunk_id,
            source_document_id=r.source_document_id,
            filename=r.original_filename,
            source_type=r.source_type,
            modality=r.modality,
            page_number=r.page_number,
            chunk_index=r.chunk_index,
            timestamp_start=r.timestamp_start,
            timestamp_end=r.timestamp_end,
            score=r.score,
            content=content,
            preview=preview,
        )

        ctx.blocks.append(block)
        ctx.total_chars += len(content)

    logger.info(
        "Context assembled: %d blocks, %d chars "
        "(dropped: %d low-score, %d dedup, %d budget)",
        len(ctx.blocks),
        ctx.total_chars,
        ctx.dropped_low_score,
        ctx.dropped_dedup,
        ctx.dropped_budget,
    )

    return ctx
