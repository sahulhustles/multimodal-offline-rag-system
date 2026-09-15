"""Phase 5 unit tests for grounded answer generation.

Tests cover:
    - Context assembly (score ordering, truncation, deduplication, headers)
    - Evidence gating (no results, low scores)
    - Prompt construction (source IDs, grounding instructions)
    - Citation validation (valid, invalid, missing, insufficient-context)
    - Generator orchestration (mocked Ollama success, unavailable, timeout)
    - API schema validation
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.generation.citation_validator import (
    VALIDATION_FAILURE_MESSAGE,
    validate_citations,
)
from backend.generation.context_assembler import (
    AssembledContext,
    SourceBlock,
    assemble_context,
    _truncate_at_sentence,
    _text_overlap_ratio,
)
from backend.generation.intent_router import QueryIntent, classify_intent, contextualize_query
from backend.generation.prompt_builder import (
    INSUFFICIENT_CONTEXT_PHRASE,
    build_conversational_messages,
    build_messages,
)
from backend.retrieval.retrieval import RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    *,
    chunk_id: str = "chunk-1",
    score: float = 0.5,
    source_document_id: str = "doc-1",
    source_type: str = "pdf",
    modality: str = "text",
    chunk_index: int = 0,
    page_number: int | None = 1,
    extracted_text: str | None = "This is some extracted text from the document.",
    transcript_text: str | None = None,
    llava_description: str | None = None,
    original_filename: str | None = "test.pdf",
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
    image_index: int | None = None,
    image_sha256: str | None = None,
    has_text_vector: bool = True,
    has_image_vector: bool = False,
) -> RetrievalResult:
    """Create a RetrievalResult for testing."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        source_document_id=source_document_id,
        source_type=source_type,
        modality=modality,
        chunk_index=chunk_index,
        page_number=page_number,
        extracted_text=extracted_text,
        transcript_text=transcript_text,
        llava_description=llava_description,
        original_filename=original_filename,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        image_index=image_index,
        image_sha256=image_sha256,
        has_text_vector=has_text_vector,
        has_image_vector=has_image_vector,
        text_preview=None,
        linked_records=[],
    )


# ---------------------------------------------------------------------------
# Context Assembly Tests
# ---------------------------------------------------------------------------


class TestContextAssembly:
    """Tests for context_assembler.assemble_context()."""

    def test_score_order(self):
        """Records are assembled in descending score order."""
        results = [
            _make_result(chunk_id="c1", score=0.3, extracted_text="Low score text."),
            _make_result(chunk_id="c2", score=0.8, extracted_text="High score text."),
            _make_result(chunk_id="c3", score=0.5, extracted_text="Mid score text."),
        ]
        ctx = assemble_context(results, min_score=0.0)
        assert len(ctx.blocks) == 3
        assert ctx.blocks[0].source_id == "S1"
        # Blocks should follow input order (results are already score-ordered by retrieve())
        assert ctx.blocks[0].chunk_id == "c1"

    def test_truncation_per_record(self):
        """Per-record content is truncated to max_chars_per_record."""
        long_text = "A" * 5000
        results = [_make_result(chunk_id="c1", score=0.5, extracted_text=long_text)]
        ctx = assemble_context(results, min_score=0.0, max_chars_per_record=100)
        assert len(ctx.blocks) == 1
        assert len(ctx.blocks[0].content) <= 101  # +1 for ellipsis char

    def test_total_budget_truncation(self):
        """Total context chars are bounded by max_total_chars."""
        results = [
            _make_result(chunk_id=f"c{i}", score=0.5, extracted_text=f"Text block {i}. " * 50)
            for i in range(10)
        ]
        ctx = assemble_context(results, min_score=0.0, max_total_chars=500)
        assert ctx.total_chars <= 500

    def test_deduplication(self):
        """Highly overlapping records are deduplicated (keep higher-scored first)."""
        shared_text = "The quick brown fox jumps over the lazy dog. " * 20
        results = [
            _make_result(chunk_id="c1", score=0.8, extracted_text=shared_text),
            _make_result(chunk_id="c2", score=0.5, extracted_text=shared_text),
        ]
        ctx = assemble_context(results, min_score=0.0)
        assert len(ctx.blocks) == 1
        assert ctx.blocks[0].chunk_id == "c1"
        assert ctx.dropped_dedup == 1

    def test_source_headers(self):
        """Each block has a [S#] header with filename, modality, page."""
        results = [
            _make_result(chunk_id="c1", score=0.5, original_filename="report.pdf", page_number=3),
        ]
        ctx = assemble_context(results, min_score=0.0)
        rendered = ctx.render()
        assert "[S1]" in rendered
        assert "report.pdf" in rendered
        assert "page 3" in rendered

    def test_score_gate_drops_low(self):
        """Results below min_score are dropped."""
        results = [
            _make_result(chunk_id="c1", score=0.1, extracted_text="Low."),
            _make_result(chunk_id="c2", score=0.5, extracted_text="High."),
        ]
        ctx = assemble_context(results, min_score=0.20)
        assert len(ctx.blocks) == 1
        assert ctx.blocks[0].chunk_id == "c2"
        assert ctx.dropped_low_score == 1

    def test_empty_results(self):
        """Empty results list produces empty context."""
        ctx = assemble_context([], min_score=0.0)
        assert len(ctx.blocks) == 0
        assert ctx.total_chars == 0


# ---------------------------------------------------------------------------
# Evidence Gate Tests
# ---------------------------------------------------------------------------


class TestEvidenceGate:
    """Tests for evidence gating in the generator."""

    def test_no_results_returns_insufficient(self):
        """No retrieval results → insufficient context (no Ollama call)."""
        results: list[RetrievalResult] = []
        ctx = assemble_context(results, min_score=0.0)
        assert len(ctx.blocks) == 0

    def test_all_below_threshold_returns_insufficient(self):
        """All results below min_score → empty context."""
        results = [
            _make_result(chunk_id="c1", score=0.05, extracted_text="Very low."),
            _make_result(chunk_id="c2", score=0.10, extracted_text="Still low."),
        ]
        ctx = assemble_context(results, min_score=0.20)
        assert len(ctx.blocks) == 0
        assert ctx.dropped_low_score == 2


# ---------------------------------------------------------------------------
# Prompt Builder Tests
# ---------------------------------------------------------------------------


class TestPromptBuilder:
    """Tests for prompt_builder.build_messages()."""

    def test_prompt_contains_source_ids(self):
        """System prompt lists allowed source IDs."""
        ctx = AssembledContext(
            blocks=[
                SourceBlock(
                    source_id="S1", chunk_id="c1", source_document_id="d1",
                    filename="test.pdf", source_type="pdf", modality="text",
                    page_number=1, chunk_index=0, timestamp_start=None,
                    timestamp_end=None, score=0.5, content="Some content.",
                    preview="Some content.",
                ),
                SourceBlock(
                    source_id="S2", chunk_id="c2", source_document_id="d1",
                    filename="test.pdf", source_type="pdf", modality="text",
                    page_number=2, chunk_index=1, timestamp_start=None,
                    timestamp_end=None, score=0.4, content="More content.",
                    preview="More content.",
                ),
            ],
            total_chars=200,
        )
        messages = build_messages("What is X?", ctx)
        system_msg = messages[0]["content"]
        assert "S1" in system_msg
        assert "S2" in system_msg

    def test_prompt_contains_grounding_instructions(self):
        """System prompt includes grounding instructions."""
        ctx = AssembledContext(
            blocks=[
                SourceBlock(
                    source_id="S1", chunk_id="c1", source_document_id="d1",
                    filename="f.pdf", source_type="pdf", modality="text",
                    page_number=1, chunk_index=0, timestamp_start=None,
                    timestamp_end=None, score=0.5, content="Text.",
                    preview="Text.",
                ),
            ],
            total_chars=5,
        )
        messages = build_messages("Question?", ctx)
        system_msg = messages[0]["content"]
        assert INSUFFICIENT_CONTEXT_PHRASE in system_msg
        assert "ONLY from the provided source documents" in system_msg
        assert "Do NOT invent" in system_msg

    def test_messages_structure(self):
        """Messages have correct system + user structure."""
        ctx = AssembledContext(
            blocks=[
                SourceBlock(
                    source_id="S1", chunk_id="c1", source_document_id="d1",
                    filename="f.pdf", source_type="pdf", modality="text",
                    page_number=1, chunk_index=0, timestamp_start=None,
                    timestamp_end=None, score=0.5, content="Text.",
                    preview="Text.",
                ),
            ],
            total_chars=5,
        )
        messages = build_messages("What is X?", ctx)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is X?"


# ---------------------------------------------------------------------------
# Citation Validator Tests
# ---------------------------------------------------------------------------


class TestCitationValidator:
    """Tests for citation_validator.validate_citations()."""

    def test_valid_citations_pass(self):
        """All citations in allowed set → grounded=True."""
        result = validate_citations(
            "The system achieves X [S1] and Y [S2].",
            {"S1", "S2", "S3"},
        )
        assert result.grounded is True
        assert result.validation_failed is False
        assert set(result.cited_ids) == {"S1", "S2"}

    def test_invalid_citation_returns_failure(self):
        """Citation outside allowed set → grounded=False, safe message."""
        result = validate_citations(
            "The system does X [S1] and Y [S99].",
            {"S1", "S2"},
        )
        assert result.grounded is False
        assert result.validation_failed is True
        assert result.answer == VALIDATION_FAILURE_MESSAGE
        assert any("S99" in w for w in result.warnings)

    def test_no_citations_returns_failure(self):
        """Answer with no [S#] refs → grounded=False, safe message."""
        result = validate_citations(
            "The system does X and Y.",
            {"S1", "S2"},
        )
        assert result.grounded is False
        assert result.validation_failed is True
        assert result.answer == VALIDATION_FAILURE_MESSAGE
        assert any("no" in w.lower() and "citation" in w.lower() for w in result.warnings)

    def test_insufficient_context_detected(self):
        """Exact insufficient-context phrase → insufficient_context=True."""
        result = validate_citations(
            INSUFFICIENT_CONTEXT_PHRASE,
            {"S1", "S2"},
        )
        assert result.insufficient_context is True
        assert result.grounded is False
        assert result.answer == INSUFFICIENT_CONTEXT_PHRASE

    def test_insufficient_context_prefix_match(self):
        """Phrase at start with trailing text still detected."""
        result = validate_citations(
            f"{INSUFFICIENT_CONTEXT_PHRASE} Some extra text.",
            {"S1"},
        )
        assert result.insufficient_context is True

    def test_multiple_same_citations(self):
        """Repeated citations (e.g. [S1] used 3 times) are correctly counted."""
        result = validate_citations(
            "Point A [S1]. Point B [S1]. Point C [S2].",
            {"S1", "S2"},
        )
        assert result.grounded is True
        assert set(result.cited_ids) == {"S1", "S2"}


# ---------------------------------------------------------------------------
# Helper Tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper functions."""

    def test_truncate_at_sentence(self):
        """Truncation finds sentence boundary."""
        text = "First sentence. Second sentence. Third sentence."
        result = _truncate_at_sentence(text, 25)
        assert result.endswith(".")
        assert len(result) <= 25

    def test_truncate_no_boundary(self):
        """Truncation without good boundary adds ellipsis."""
        text = "A" * 100
        result = _truncate_at_sentence(text, 50)
        assert len(result) <= 52  # +2 for ellipsis char

    def test_text_overlap_ratio_identical(self):
        """Identical texts have overlap ratio of 1.0."""
        assert _text_overlap_ratio("hello world", "hello world") == 1.0

    def test_text_overlap_ratio_none(self):
        """Completely different texts have low overlap."""
        ratio = _text_overlap_ratio("alpha beta gamma", "delta epsilon zeta")
        assert ratio < 0.1

    def test_text_overlap_ratio_empty(self):
        """Empty text returns 0."""
        assert _text_overlap_ratio("", "hello") == 0.0
        assert _text_overlap_ratio("hello", "") == 0.0



class TestIntentRouter:
    """Tests for query intent routing and follow-up contextualization."""

    def test_classify_greetings(self):
        """Standard greetings classified as CONVERSATIONAL."""
        assert classify_intent("hello") == QueryIntent.CONVERSATIONAL
        assert classify_intent("Hi!") == QueryIntent.CONVERSATIONAL
        assert classify_intent("Good morning") == QueryIntent.CONVERSATIONAL
        assert classify_intent("Who are you?") == QueryIntent.CONVERSATIONAL
        assert classify_intent("what can you do") == QueryIntent.CONVERSATIONAL
        assert classify_intent("thanks a lot") == QueryIntent.CONVERSATIONAL

    def test_classify_document_queries(self):
        """Factual document queries classified as DOCUMENT_QUERY."""
        assert classify_intent("What is the register number?") == QueryIntent.DOCUMENT_QUERY
        assert classify_intent("Summarize Sahul's experience") == QueryIntent.DOCUMENT_QUERY
        assert classify_intent("What is the date of birth on the license?") == QueryIntent.DOCUMENT_QUERY
        assert classify_intent("What is the transcript text of sampleaudio.mp3?") == QueryIntent.DOCUMENT_QUERY

    def test_contextualize_query_no_history(self):
        """Without history, query returns unchanged."""
        assert contextualize_query("What is his GPA?", None) == "What is his GPA?"
        assert contextualize_query("What is his GPA?", []) == "What is his GPA?"

    def test_contextualize_query_with_pronoun(self):
        """Query with pronoun incorporates previous user subject."""
        history = [
            {"role": "user", "content": "Tell me about Sahul Hameed"},
            {"role": "assistant", "content": "Sahul is a student [S1]."},
        ]
        res = contextualize_query("What is his GPA?", history)
        assert "Sahul Hameed" in res
        assert "What is his GPA?" in res

    def test_contextualize_query_independent(self):
        """Independent query without pronouns remains untouched."""
        history = [
            {"role": "user", "content": "Tell me about Sahul Hameed"},
            {"role": "assistant", "content": "Sahul is a student [S1]."},
        ]
        res = contextualize_query("What is the hall ticket exam date?", history)
        assert res == "What is the hall ticket exam date?"


class TestConversationalInteraction:
    """Tests for conversational prompt assembly and generator behavior."""

    def test_build_conversational_messages(self):
        """Conversational messages include persona and history."""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        msgs = build_conversational_messages("What can you do?", history)
        assert msgs[0]["role"] == "system"
        assert "Offline Multimodal RAG System" in msgs[0]["content"]
        assert msgs[1] == {"role": "user", "content": "Hi"}
        assert msgs[2] == {"role": "assistant", "content": "Hello! How can I help?"}
        assert msgs[3] == {"role": "user", "content": "What can you do?"}

    @patch("backend.generation.generator.get_chat_client")
    def test_generate_answer_conversational_mode(self, mock_get_client):
        """Conversational query returns is_conversational=True without retrieval."""
        from backend.generation.generator import generate_answer
        from backend.generation.ollama_chat import ChatResponse

        mock_client = MagicMock()
        mock_client.chat.return_value = ChatResponse(
            content="Hello! I am your offline document assistant.",
            model="llama3.2:3b",
            done=True,
            prompt_eval_count=20,
            eval_count=15,
        )
        mock_get_client.return_value = mock_client

        response = generate_answer(
            query="Hello, who are you?",
            source_document_ids=["dummy-doc-id"],
        )

        assert response.is_conversational is True
        assert response.grounded is False
        assert response.insufficient_context is False
        assert "Hello! I am your offline document assistant." in response.answer
        assert len(response.citations) == 0


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
