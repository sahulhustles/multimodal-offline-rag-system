"""Verification test for Phase 3: Qdrant Indexing Engine.

Tests:
  1. Collection setup & payload indexes verification
  2. Deterministic UUIDv5 ID generation & stability
  3. Text chunk indexing (embeddings, payload preview, SQLite IndexedRecord)
  4. Idempotent re-indexing (point IDs remain stable, no duplicate points)
  5. Index status reporting
  6. Safe deletion cleanup (SQLite records + Qdrant points removed)

Run:
    python -m backend.tests.test_phase3_indexing
"""

import sys
import uuid
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlmodel import select

from backend.config import settings
from backend.db.database import get_session, init_db
from backend.db.models import (
    DocumentSourceType,
    IndexedRecord,
    IndexingStatus,
    IngestionJob,
    JobStatus,
    Modality,
    RecordSourceType,
    SourceDocument,
)
from backend.indexing.indexer import (
    delete_indexed_records,
    get_index_status,
    index_image,
    index_text_chunks,
    make_clip_id,
    make_llava_desc_id,
    make_text_chunk_id,
)
from backend.indexing.qdrant_manager import (
    count_points_by_source,
    ensure_collection,
    ensure_payload_indexes,
    get_qdrant_client,
)
from backend.processors.schemas import (
    ClipEmbeddingResult,
    TextChunkResult,
    TextEmbeddingResult,
    VisionProcessingResult,
)

PASS = "[PASS]"
FAIL = "[FAIL]"


def test_collection_and_payload_indexes():
    """Test 1: Verify Qdrant collection setup and payload indexes."""
    print(f"\n{'='*60}")
    print("Test 1: Collection & Payload Indexes")

    client = get_qdrant_client()
    ensure_collection()
    ensure_payload_indexes()

    info = client.get_collection(settings.qdrant_collection_name)
    schema = info.payload_schema or {}
    expected_indexes = ["source_document_id", "modality", "source_type", "indexing_status"]

    all_found = True
    for key in expected_indexes:
        found = key in schema
        print(f"  Payload index '{key}': {'EXISTS' if found else 'MISSING'}")
        if not found:
            all_found = False

    ok = info.status.value in ["green", "yellow"] and all_found
    print(f"  Result: {PASS if ok else FAIL}")
    return ok


def test_deterministic_point_ids():
    """Test 2: Deterministic UUIDv5 point ID generation."""
    print(f"\n{'='*60}")
    print("Test 2: Deterministic UUIDv5 Generation")

    doc_id = "test-doc-12345678-abcd-ef01-2345-6789abcdef01"

    id1 = make_text_chunk_id(doc_id, 0)
    id2 = make_text_chunk_id(doc_id, 0)
    id_diff = make_text_chunk_id(doc_id, 1)

    is_stable = id1 == id2
    is_valid_uuid = False
    try:
        uuid_obj = uuid.UUID(id1)
        is_valid_uuid = uuid_obj.version == 5
    except Exception:
        pass

    # Occurrence-aware image IDs (Phase 3 bugfix)
    # The same image SHA-256 appearing on Page 1 (img 0) vs Page 2 (img 1)
    # must produce distinct UUIDv5 IDs to prevent SQLite UNIQUE constraint failure.
    shared_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    clip_p1_img0 = make_clip_id(doc_id, shared_hash, page_number=1, image_index=0)
    clip_p1_img0_repeat = make_clip_id(doc_id, shared_hash, page_number=1, image_index=0)
    clip_p2_img1 = make_clip_id(doc_id, shared_hash, page_number=2, image_index=1)

    desc_p1_img0 = make_llava_desc_id(doc_id, shared_hash, page_number=1, image_index=0)
    desc_p2_img1 = make_llava_desc_id(doc_id, shared_hash, page_number=2, image_index=1)

    clip_stable = clip_p1_img0 == clip_p1_img0_repeat
    clip_distinct_across_occurrences = clip_p1_img0 != clip_p2_img1
    desc_distinct_across_occurrences = desc_p1_img0 != desc_p2_img1
    clip_diff_from_desc = clip_p1_img0 != desc_p1_img0
    all_distinct = len({id1, id_diff, clip_p1_img0, clip_p2_img1, desc_p1_img0, desc_p2_img1}) == 6

    print(f"  Text ID 1: {id1}")
    print(f"  CLIP Page 1 (img 0): {clip_p1_img0}")
    print(f"  CLIP Page 2 (img 1): {clip_p2_img1}")
    print(f"  Stable (id1 == id2)                  : {is_stable}")
    print(f"  CLIP Stable on re-compute            : {clip_stable}")
    print(f"  CLIP Distinct across occurrences     : {clip_distinct_across_occurrences}")
    print(f"  LLaVA Distinct across occurrences    : {desc_distinct_across_occurrences}")
    print(f"  CLIP != LLaVA desc for same image    : {clip_diff_from_desc}")
    print(f"  UUIDv5 Valid                         : {is_valid_uuid}")

    ok = (
        is_stable
        and clip_stable
        and clip_distinct_across_occurrences
        and desc_distinct_across_occurrences
        and clip_diff_from_desc
        and all_distinct
        and is_valid_uuid
    )
    print(f"  Result: {PASS if ok else FAIL}")
    return ok


def test_text_chunk_indexing():
    """Test 3: Index text chunks into Qdrant & SQLite."""
    print(f"\n{'='*60}")
    print("Test 3: Text Chunk Indexing")

    init_db()
    test_job_id = f"job-{uuid.uuid4().hex[:8]}"
    test_doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    filename = "phase3_test_document.docx"

    # 1. Create a dummy IngestionJob and SourceDocument in SQLite
    with get_session() as db:
        job = IngestionJob(id=test_job_id, status=JobStatus.completed)
        db.add(job)
        db.commit()

        doc = SourceDocument(
            source_document_id=test_doc_id,
            job_id=test_job_id,
            original_filename=filename,
            source_type=DocumentSourceType.docx,
            file_size_bytes=1024,
            ingestion_status=JobStatus.completed,
            indexing_status=IndexingStatus.not_started,
        )
        db.add(doc)
        db.commit()

    # 2. Create sample TextChunkResult objects
    chunks = [
        TextChunkResult(
            source_document_id=test_doc_id,
            chunk_index=0,
            text="Antigravity Phase 3 implements Qdrant vector indexing and point persistence.",
            token_count=12,
            page_number=1,
        ),
        TextChunkResult(
            source_document_id=test_doc_id,
            chunk_index=1,
            text="All points use deterministic UUIDv5 identifiers to guarantee idempotency on re-index.",
            token_count=14,
            page_number=1,
        ),
    ]

    # 3. Index chunks
    result = index_text_chunks(chunks, test_doc_id, RecordSourceType.docx, filename)
    indexed_count = result.get("indexed", 0)
    print(f"  Indexed chunks count: {indexed_count}")
    # 4. Verify in Qdrant
    qdrant_points = count_points_by_source(test_doc_id)
    print(f"  Qdrant points found : {qdrant_points}")

    # 5. Verify in SQLite
    with get_session() as db:
        records = db.exec(select(IndexedRecord).where(IndexedRecord.source_document_id == test_doc_id)).all()
        db_records_count = len(records)
        print(f"  SQLite records found: {db_records_count}")

        # Check preview vs full text
        has_text = records[0].extracted_text is not None and len(records[0].extracted_text) > 0
        point_id_valid = records[0].qdrant_point_id is not None

    ok = indexed_count == 2 and qdrant_points == 2 and db_records_count == 2 and has_text and point_id_valid
    print(f"  Result: {PASS if ok else FAIL}")
    return ok, test_doc_id, test_job_id, chunks, filename


def test_idempotent_reindexing(test_doc_id: str, chunks: list, filename: str):
    """Test 4: Re-indexing does not duplicate points and IDs stay stable."""
    print(f"\n{'='*60}")
    print("Test 4: Idempotent Re-indexing")

    # Capture point IDs before
    with get_session() as db:
        records_before = db.exec(select(IndexedRecord).where(IndexedRecord.source_document_id == test_doc_id)).all()
        ids_before = {r.chunk_id: r.qdrant_point_id for r in records_before}

    # Delete existing and re-index (simulating re-indexing flow)
    delete_indexed_records(test_doc_id)
    index_text_chunks(chunks, test_doc_id, RecordSourceType.docx, filename)

    # Capture point IDs after
    with get_session() as db:
        records_after = db.exec(select(IndexedRecord).where(IndexedRecord.source_document_id == test_doc_id)).all()
        ids_after = {r.chunk_id: r.qdrant_point_id for r in records_after}

    points_after = count_points_by_source(test_doc_id)

    ids_identical = ids_before == ids_after
    no_duplicates = points_after == 2 and len(records_after) == 2

    print(f"  Points after re-index: {points_after} (expected 2)")
    print(f"  Point IDs identical   : {ids_identical}")
    print(f"  No duplicate records  : {no_duplicates}")

    ok = ids_identical and no_duplicates
    print(f"  Result: {PASS if ok else FAIL}")
    return ok


def test_index_status_and_cleanup(test_doc_id: str, test_job_id: str):
    """Test 5 & 6: Index status reporting and complete deletion."""
    print(f"\n{'='*60}")
    print("Test 5 & 6: Status & Cleanup")

    status = get_index_status(test_doc_id)
    print(f"  Indexed records count in status: {status['total_indexed_records']}")
    print(f"  Qdrant points count in status : {status['qdrant_point_count']}")
    print(f"  Breakdown by modality         : {status['by_modality']}")

    status_ok = status["total_indexed_records"] == 2 and status["qdrant_point_count"] == 2

    # Cleanup test document and points
    deleted = delete_indexed_records(test_doc_id)
    points_remaining = count_points_by_source(test_doc_id)

    with get_session() as db:
        records_remaining = len(db.exec(select(IndexedRecord).where(IndexedRecord.source_document_id == test_doc_id)).all())
        # Also clean test doc and job
        for doc in db.exec(select(SourceDocument).where(SourceDocument.source_document_id == test_doc_id)).all():
            db.delete(doc)
        for job in db.exec(select(IngestionJob).where(IngestionJob.id == test_job_id)).all():
            db.delete(job)
        db.commit()

    print(f"  Points remaining after cleanup : {points_remaining} (expected 0)")
    print(f"  Records remaining after cleanup: {records_remaining} (expected 0)")

    cleanup_ok = points_remaining == 0 and records_remaining == 0 and deleted["deleted_sqlite"] == 2
    ok = status_ok and cleanup_ok
    print(f"  Result: {PASS if ok else FAIL}")
    return ok


def test_repeated_image_indexing():
    """Test 6: Repeated images (same sha256) across pages use occurrence-aware IDs and avoid UNIQUE constraint."""
    print(f"\n{'='*60}")
    print("Test 6: Repeated Image Indexing (Occurrence-Aware)")

    init_db()
    test_job_id = f"job-{uuid.uuid4().hex[:8]}"
    test_doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    filename = "repeated_logos_test.pdf"

    with get_session() as db:
        job = IngestionJob(id=test_job_id, status=JobStatus.completed)
        db.add(job)
        doc = SourceDocument(
            source_document_id=test_doc_id,
            job_id=test_job_id,
            original_filename=filename,
            source_type=DocumentSourceType.pdf,
            file_size_bytes=4096,
            ingestion_status=JobStatus.completed,
            indexing_status=IndexingStatus.not_started,
        )
        db.add(doc)
        db.commit()

    shared_hash = "d41d8cd98f00b204e9800998ecf8427e"

    # Simulate 2 occurrences of the same embedded logo on page 1 and page 2
    def make_mock_vision_result(page: int, idx: int) -> VisionProcessingResult:
        clip_emb = ClipEmbeddingResult(
            vector=[0.05] * 512,
            vector_dimension=512,
            embedding_model="clip-ViT-B-32",
            image_path=f"page_{page}_img_{idx}.png",
            preprocessing="resize",
        )
        text_emb = TextEmbeddingResult(
            vector=[0.02] * 384,
            vector_dimension=384,
            embedding_model="all-MiniLM-L6-v2",
            embedding_span_count=1,
            source_text=f"Header logo on page {page}",
            token_count=5,
        )
        return VisionProcessingResult(
            source_document_id=test_doc_id,
            image_path=f"page_{page}_img_{idx}.png",
            original_filename=f"logo_page{page}.png",
            clip_embedding=clip_emb,
            clip_status="completed",
            llava_description=f"Header logo on page {page}",
            llava_description_embedding=text_emb,
            llava_status="completed",
            ingestion_status="completed",
            page_number=page,
            image_hash=shared_hash,
        )

    vis_p1 = make_mock_vision_result(page=1, idx=0)
    vis_p2 = make_mock_vision_result(page=2, idx=1)

    # Index occurrence 0 (page 1)
    res1 = index_image(
        vis_p1, test_doc_id,
        clip_source_type=RecordSourceType.pdf_extracted_image,
        desc_source_type=RecordSourceType.pdf_image_description,
        original_filename=vis_p1.original_filename,
        image_index=0,
    )
    # Index occurrence 1 (page 2) - same SHA256!
    res2 = index_image(
        vis_p2, test_doc_id,
        clip_source_type=RecordSourceType.pdf_extracted_image,
        desc_source_type=RecordSourceType.pdf_image_description,
        original_filename=vis_p2.original_filename,
        image_index=1,
    )

    qdrant_points = count_points_by_source(test_doc_id)
    print(f"  Qdrant points for repeated image: {qdrant_points} (expected 4)")

    with get_session() as db:
        records = db.exec(
            select(IndexedRecord).where(IndexedRecord.source_document_id == test_doc_id)
        ).all()
        sqlite_count = len(records)
        chunk_ids = [r.chunk_id for r in records]
        unique_chunk_ids = set(chunk_ids)
        indices_present = {r.image_index for r in records}
        hashes_present = {r.image_sha256 for r in records}

    print(f"  SQLite records for repeated image: {sqlite_count} (expected 4)")
    print(f"  Unique chunk IDs count          : {len(unique_chunk_ids)} (expected 4)")
    print(f"  Occurrence indices present      : {indices_present} (expected {{0, 1}})")
    print(f"  SHA-256 hashes present          : {hashes_present}")

    no_collision = sqlite_count == 4 and len(unique_chunk_ids) == 4 and qdrant_points == 4
    metadata_correct = indices_present == {0, 1} and hashes_present == {shared_hash}

    # Safe cleanup
    delete_indexed_records(test_doc_id)
    with get_session() as db:
        for doc in db.exec(select(SourceDocument).where(SourceDocument.source_document_id == test_doc_id)).all():
            db.delete(doc)
        for job in db.exec(select(IngestionJob).where(IngestionJob.id == test_job_id)).all():
            db.delete(job)
        db.commit()

    ok = no_collision and metadata_correct
    print(f"  Result: {PASS if ok else FAIL}")
    return ok


def main():
    print(f"{'='*60}")
    print("Starting Phase 3 Verification Test Suite")
    print(f"{'='*60}")

    results = []

    # 1. Collection & Payload schema
    r1 = test_collection_and_payload_indexes()
    results.append(("Collection & Payload Schema", r1))

    # 2. Deterministic UUIDv5 Point IDs
    r2 = test_deterministic_point_ids()
    results.append(("Deterministic UUIDv5 Point IDs", r2))

    # 3. Text Chunk Indexing
    r3, doc_id, job_id, chunks, filename = test_text_chunk_indexing()
    results.append(("Text Chunk Indexing", r3))

    # 4. Idempotent Re-indexing
    r4 = test_idempotent_reindexing(doc_id, chunks, filename)
    results.append(("Idempotent Re-indexing", r4))

    # 5. Status Reporting & Safe Cleanup
    r5 = test_index_status_and_cleanup(doc_id, job_id)
    results.append(("Status Reporting & Safe Cleanup", r5))

    # 6. Repeated Image Indexing (Occurrence-Aware Fix)
    r6 = test_repeated_image_indexing()
    results.append(("Repeated Image Indexing (Bugfix)", r6))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for name, passed in results:
        status_str = PASS if passed else FAIL
        print(f"  {name:35s}: {status_str}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\nALL PHASE 3 TESTS PASSED SUCCESSFULLY! ({len(results)}/{len(results)})")
    else:
        print(f"\nSOME TESTS FAILED! ({len(results)} tests)")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
