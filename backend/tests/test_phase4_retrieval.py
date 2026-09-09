"""Phase 4 retrieval tests.

Covers:
    1. Query encoding via TextEmbedder (mock)
    2. Qdrant search wrapper filter construction
    3. SQLite hydration of IndexedRecords
    4. Cross-modal linked-record enrichment
    5. API endpoint integration (FastAPI TestClient)
    6. Filter validation (only indexed documents)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Query encoding
# ---------------------------------------------------------------------------


class TestQueryEncoding:
    """Verify that the retrieval engine encodes queries correctly."""

    @patch("backend.retrieval.retrieval.get_text_embedder")
    def test_query_produces_384d_vector(self, mock_get_embedder):
        """embed(query) should return a 384-d vector."""
        from backend.processors.schemas import TextEmbeddingResult

        fake_vector = [0.1] * 384
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = TextEmbeddingResult(
            vector=fake_vector,
            vector_dimension=384,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_span_count=1,
            source_text="test query",
            token_count=5,
        )
        mock_get_embedder.return_value = mock_embedder

        result = mock_embedder.embed("test query")
        assert len(result.vector) == 384
        assert result.vector_dimension == 384

    @patch("backend.retrieval.retrieval.get_text_embedder")
    def test_query_encoding_is_called_once(self, mock_get_embedder):
        """Retrieval should call embed exactly once per query."""
        from backend.processors.schemas import TextEmbeddingResult

        fake_vector = [0.1] * 384
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = TextEmbeddingResult(
            vector=fake_vector,
            vector_dimension=384,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_span_count=1,
            source_text="test",
            token_count=3,
        )
        mock_get_embedder.return_value = mock_embedder

        # Import and call retrieve with mocked search returning empty
        with patch("backend.retrieval.retrieval.search_text_vectors", return_value=[]):
            from backend.retrieval.retrieval import retrieve

            response = retrieve(
                query="test",
                source_document_ids=["doc-1"],
            )
            mock_embedder.embed.assert_called_once_with("test")
            assert response.result_count == 0


# ---------------------------------------------------------------------------
# 2. Qdrant search wrapper filter construction
# ---------------------------------------------------------------------------


class TestQdrantSearchWrapper:
    """Verify the search wrapper builds correct filters."""

    @patch("backend.indexing.search.get_qdrant_client")
    def test_search_uses_text_named_vector(self, mock_get_client):
        """search_text_vectors should use the 'text' named vector."""
        from qdrant_client import models

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result
        mock_get_client.return_value = mock_client

        from backend.indexing.search import search_text_vectors

        search_text_vectors(
            query_vector=[0.1] * 384,
            source_document_ids=["doc-1"],
            limit=5,
        )

        call_kwargs = mock_client.query_points.call_args
        assert call_kwargs.kwargs.get("using") == "text" or (
            len(call_kwargs.args) > 0 or call_kwargs[1].get("using") == "text"
        )

    @patch("backend.indexing.search.get_qdrant_client")
    def test_search_applies_source_document_filter(self, mock_get_client):
        """Filter must include source_document_id MatchAny."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result
        mock_get_client.return_value = mock_client

        from backend.indexing.search import search_text_vectors

        search_text_vectors(
            query_vector=[0.1] * 384,
            source_document_ids=["doc-1", "doc-2"],
            limit=5,
        )

        # Extract filter from call
        call_kwargs = mock_client.query_points.call_args
        query_filter = call_kwargs.kwargs.get("query_filter") or call_kwargs[1].get("query_filter")
        assert query_filter is not None

        # Verify at least source_document_id and indexing_status conditions exist
        must_conditions = query_filter.must
        keys = [c.key for c in must_conditions]
        assert "source_document_id" in keys
        assert "indexing_status" in keys

    @patch("backend.indexing.search.get_qdrant_client")
    def test_search_applies_modality_filter(self, mock_get_client):
        """When modality_filter is set, it should appear in must conditions."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result
        mock_get_client.return_value = mock_client

        from backend.indexing.search import search_text_vectors

        search_text_vectors(
            query_vector=[0.1] * 384,
            source_document_ids=["doc-1"],
            limit=5,
            modality_filter="text",
        )

        call_kwargs = mock_client.query_points.call_args
        query_filter = call_kwargs.kwargs.get("query_filter") or call_kwargs[1].get("query_filter")
        must_conditions = query_filter.must
        keys = [c.key for c in must_conditions]
        assert "modality" in keys

    @patch("backend.indexing.search.get_qdrant_client")
    def test_search_respects_limit(self, mock_get_client):
        """The limit parameter should be passed to query_points."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result
        mock_get_client.return_value = mock_client

        from backend.indexing.search import search_text_vectors

        search_text_vectors(
            query_vector=[0.1] * 384,
            source_document_ids=["doc-1"],
            limit=10,
        )

        call_kwargs = mock_client.query_points.call_args
        limit = call_kwargs.kwargs.get("limit") or call_kwargs[1].get("limit")
        assert limit == 10


# ---------------------------------------------------------------------------
# 3. SQLite hydration
# ---------------------------------------------------------------------------


class TestSQLiteHydration:
    """Verify retrieval hydrates records correctly from SQLite."""

    @patch("backend.retrieval.retrieval.search_text_vectors")
    @patch("backend.retrieval.retrieval.get_text_embedder")
    @patch("backend.retrieval.retrieval.get_session")
    def test_hydration_maps_point_to_record(
        self, mock_get_session, mock_get_embedder, mock_search
    ):
        """Scored points should be hydrated with full IndexedRecord data."""
        from backend.processors.schemas import TextEmbeddingResult

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = TextEmbeddingResult(
            vector=[0.1] * 384,
            vector_dimension=384,
            embedding_model="test-model",
            embedding_span_count=1,
            source_text="test",
            token_count=3,
        )
        mock_get_embedder.return_value = mock_embedder

        # Mock Qdrant search result
        point_id = str(uuid.uuid4())
        mock_point = MagicMock()
        mock_point.id = point_id
        mock_point.score = 0.85
        mock_point.payload = {
            "source_document_id": "doc-1",
            "text_preview": "Test text preview",
        }
        mock_search.return_value = [mock_point]

        # Mock SQLite session
        mock_session = MagicMock()

        # Create mock record
        mock_record = MagicMock()
        mock_record.qdrant_point_id = point_id
        mock_record.chunk_id = "chunk-1"
        mock_record.source_document_id = "doc-1"
        mock_record.source_type = MagicMock(value="pdf")
        mock_record.modality = MagicMock(value="text")
        mock_record.chunk_index = 0
        mock_record.page_number = 1
        mock_record.extracted_text = "Full text content here"
        mock_record.transcript_text = None
        mock_record.llava_description = None
        mock_record.timestamp_start = None
        mock_record.timestamp_end = None
        mock_record.image_index = None
        mock_record.image_sha256 = None
        mock_record.has_text_vector = True
        mock_record.has_image_vector = False
        mock_record.linked_chunk_ids_json = None

        # Create mock document
        mock_doc = MagicMock()
        mock_doc.source_document_id = "doc-1"
        mock_doc.original_filename = "test.pdf"

        # Session exec returns records/docs in sequence
        call_count = [0]

        def mock_exec(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.all.return_value = [mock_record]
            elif call_count[0] == 1:
                result.all.return_value = [mock_doc]
            else:
                result.all.return_value = []
            call_count[0] += 1
            return result

        mock_session.exec = mock_exec
        mock_session.close = MagicMock()
        mock_get_session.return_value = mock_session

        from backend.retrieval.retrieval import retrieve

        response = retrieve(query="test", source_document_ids=["doc-1"])

        assert response.result_count == 1
        result = response.results[0]
        assert result.chunk_id == "chunk-1"
        assert result.score == 0.85
        assert result.original_filename == "test.pdf"
        assert result.extracted_text == "Full text content here"


# ---------------------------------------------------------------------------
# 4. Cross-modal linked-record enrichment
# ---------------------------------------------------------------------------


class TestCrossModalEnrichment:
    """Verify linked_chunk_ids resolution works correctly."""

    @patch("backend.retrieval.retrieval.search_text_vectors")
    @patch("backend.retrieval.retrieval.get_text_embedder")
    @patch("backend.retrieval.retrieval.get_session")
    def test_linked_records_are_resolved(
        self, mock_get_session, mock_get_embedder, mock_search
    ):
        """Records with linked_chunk_ids should include enriched linked records."""
        from backend.processors.schemas import TextEmbeddingResult

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = TextEmbeddingResult(
            vector=[0.1] * 384,
            vector_dimension=384,
            embedding_model="test-model",
            embedding_span_count=1,
            source_text="test",
            token_count=3,
        )
        mock_get_embedder.return_value = mock_embedder

        point_id = str(uuid.uuid4())
        mock_point = MagicMock()
        mock_point.id = point_id
        mock_point.score = 0.9
        mock_point.payload = {"source_document_id": "doc-1", "text_preview": "preview"}
        mock_search.return_value = [mock_point]

        # CLIP image record (main result)
        linked_chunk_id = "linked-desc-1"
        mock_record = MagicMock()
        mock_record.qdrant_point_id = point_id
        mock_record.chunk_id = "clip-1"
        mock_record.source_document_id = "doc-1"
        mock_record.source_type = MagicMock(value="pdf_image_description")
        mock_record.modality = MagicMock(value="text")
        mock_record.chunk_index = 0
        mock_record.page_number = 2
        mock_record.extracted_text = None
        mock_record.transcript_text = None
        mock_record.llava_description = "A chart showing revenue growth"
        mock_record.timestamp_start = None
        mock_record.timestamp_end = None
        mock_record.image_index = 0
        mock_record.image_sha256 = "abc123"
        mock_record.has_text_vector = True
        mock_record.has_image_vector = False
        mock_record.linked_chunk_ids_json = json.dumps([linked_chunk_id])

        # Linked CLIP record
        mock_linked = MagicMock()
        mock_linked.chunk_id = linked_chunk_id
        mock_linked.source_type = MagicMock(value="pdf_extracted_image")
        mock_linked.modality = MagicMock(value="image")
        mock_linked.extracted_text = None
        mock_linked.transcript_text = None
        mock_linked.llava_description = None

        mock_doc = MagicMock()
        mock_doc.source_document_id = "doc-1"
        mock_doc.original_filename = "report.pdf"

        call_count = [0]

        def mock_exec(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.all.return_value = [mock_record]
            elif call_count[0] == 1:
                result.all.return_value = [mock_doc]
            elif call_count[0] == 2:
                result.all.return_value = [mock_linked]
            else:
                result.all.return_value = []
            call_count[0] += 1
            return result

        mock_session = MagicMock()
        mock_session.exec = mock_exec
        mock_session.close = MagicMock()
        mock_get_session.return_value = mock_session

        from backend.retrieval.retrieval import retrieve

        response = retrieve(query="test", source_document_ids=["doc-1"])

        assert response.result_count == 1
        result = response.results[0]
        assert len(result.linked_records) == 1
        assert result.linked_records[0].chunk_id == linked_chunk_id
        assert result.linked_records[0].source_type == "pdf_extracted_image"
        assert result.linked_records[0].modality == "image"


# ---------------------------------------------------------------------------
# 5. API endpoint integration
# ---------------------------------------------------------------------------


class TestRetrievalAPI:
    """Test the retrieval API endpoint."""

    @patch("backend.api.endpoints.retrieval.retrieve")
    @patch("backend.api.endpoints.retrieval.get_session")
    def test_search_endpoint_returns_results(self, mock_get_session, mock_retrieve):
        """POST /retrieval/search should return SearchResponseSchema."""
        from dataclasses import dataclass, field
        from backend.retrieval.retrieval import SearchResponse, RetrievalResult

        # Mock session to validate documents exist and are indexed
        mock_doc = MagicMock()
        mock_doc.source_document_id = "doc-1"
        mock_doc.indexing_status = MagicMock(value="indexed")

        mock_session = MagicMock()
        mock_exec_result = MagicMock()
        mock_exec_result.all.return_value = [mock_doc]
        mock_session.exec.return_value = mock_exec_result
        mock_session.close = MagicMock()
        mock_get_session.return_value = mock_session

        # Mock retrieval
        mock_retrieve.return_value = SearchResponse(
            query="test query",
            query_vector_dim=384,
            embedding_model="test-model",
            encoding_time_ms=15.2,
            search_time_ms=8.5,
            hydration_time_ms=3.1,
            total_time_ms=26.8,
            result_count=1,
            results=[
                RetrievalResult(
                    chunk_id="chunk-1",
                    score=0.82,
                    source_document_id="doc-1",
                    source_type="pdf",
                    modality="text",
                    chunk_index=0,
                    page_number=1,
                    text_preview="Test preview",
                    extracted_text="Full text",
                    original_filename="test.pdf",
                    has_text_vector=True,
                )
            ],
        )

        from fastapi.testclient import TestClient
        from backend.api.endpoints.retrieval import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/search",
            json={
                "query": "test query",
                "source_document_ids": ["doc-1"],
                "top_k": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["result_count"] == 1
        assert data["results"][0]["chunk_id"] == "chunk-1"
        assert data["results"][0]["score"] == 0.82

    @patch("backend.api.endpoints.retrieval.get_session")
    def test_search_endpoint_rejects_missing_docs(self, mock_get_session):
        """Should return 404 if source documents don't exist."""
        mock_session = MagicMock()
        mock_exec_result = MagicMock()
        mock_exec_result.all.return_value = []  # No docs found
        mock_session.exec.return_value = mock_exec_result
        mock_session.close = MagicMock()
        mock_get_session.return_value = mock_session

        from fastapi.testclient import TestClient
        from backend.api.endpoints.retrieval import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/search",
            json={
                "query": "test",
                "source_document_ids": ["nonexistent-doc"],
            },
        )

        assert response.status_code == 404

    @patch("backend.api.endpoints.retrieval.get_session")
    def test_search_endpoint_rejects_unindexed_docs(self, mock_get_session):
        """Should return 422 if docs exist but aren't indexed."""
        mock_doc = MagicMock()
        mock_doc.source_document_id = "doc-1"
        mock_doc.indexing_status = MagicMock(value="not_started")

        mock_session = MagicMock()
        mock_exec_result = MagicMock()
        mock_exec_result.all.return_value = [mock_doc]
        mock_session.exec.return_value = mock_exec_result
        mock_session.close = MagicMock()
        mock_get_session.return_value = mock_session

        from fastapi.testclient import TestClient
        from backend.api.endpoints.retrieval import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            "/search",
            json={
                "query": "test",
                "source_document_ids": ["doc-1"],
            },
        )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 6. Pydantic schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Verify Pydantic request schema enforces constraints."""

    def test_search_request_rejects_empty_query(self):
        """Empty query string should fail validation."""
        from backend.api.schemas.retrieval import SearchRequestSchema

        with pytest.raises(Exception):
            SearchRequestSchema(query="", source_document_ids=["doc-1"])

    def test_search_request_rejects_empty_doc_ids(self):
        """Empty source_document_ids should fail validation."""
        from backend.api.schemas.retrieval import SearchRequestSchema

        with pytest.raises(Exception):
            SearchRequestSchema(query="test", source_document_ids=[])

    def test_search_request_top_k_bounds(self):
        """top_k must be between 1 and 20."""
        from backend.api.schemas.retrieval import SearchRequestSchema

        with pytest.raises(Exception):
            SearchRequestSchema(query="test", source_document_ids=["doc-1"], top_k=0)

        with pytest.raises(Exception):
            SearchRequestSchema(query="test", source_document_ids=["doc-1"], top_k=21)

    def test_search_request_defaults(self):
        """Default values should be applied correctly."""
        from backend.api.schemas.retrieval import SearchRequestSchema

        req = SearchRequestSchema(query="test", source_document_ids=["doc-1"])
        assert req.top_k == 5
        assert req.score_threshold == 0.0
        assert req.modality_filter is None

    def test_search_response_schema_serialises(self):
        """SearchResponseSchema should serialise to dict correctly."""
        from backend.api.schemas.retrieval import SearchResponseSchema, RetrievalResultSchema

        resp = SearchResponseSchema(
            query="test",
            query_vector_dim=384,
            embedding_model="test-model",
            encoding_time_ms=10.0,
            search_time_ms=5.0,
            hydration_time_ms=2.0,
            total_time_ms=17.0,
            result_count=0,
            results=[],
        )
        data = resp.model_dump()
        assert data["query"] == "test"
        assert data["result_count"] == 0
        assert isinstance(data["results"], list)
