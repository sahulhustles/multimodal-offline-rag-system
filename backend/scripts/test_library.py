import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
import json

from backend.db.database import init_db, get_session
from backend.db.models import SourceDocument, IndexedRecord, DocumentSourceType, JobStatus
from backend.api.endpoints.demo import process_document_bg, DEMO_UPLOADS, DEMO_PROCESSED
from sqlmodel import select

def run_test():
    print("========================================")
    print(" Library Processing Pipeline Test")
    print("========================================")
    
    # 1. Initialize SQLite Database
    init_db()
    
    # Ensure demo directories exist
    DEMO_UPLOADS.mkdir(parents=True, exist_ok=True)
    DEMO_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # 2. Create mock note file
    doc_id = str(uuid.uuid4())
    filename = "test_note.txt"
    note_path = DEMO_UPLOADS / f"{uuid.uuid4()}_{filename}"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("This is a mock text note to verify the new local persistence pipeline in Phase 2. "
                "It should be chunked into sliding windows, and its metadata saved to a JSON file. "
                "Importantly, no IndexedRecord rows should be written to SQLite database.")
                
    print(f"Created mock file at: {note_path}")
    
    # 3. Create SourceDocument in SQLite
    with get_session() as session:
        doc = SourceDocument(
            source_document_id=doc_id,
            job_id=str(uuid.uuid4()),
            original_filename=filename,
            source_type=DocumentSourceType.text_note,
            file_path=str(note_path),
            file_size_bytes=note_path.stat().st_size,
            ingestion_status=JobStatus.queued,
            created_at=datetime.now(timezone.utc)
        )
        session.add(doc)
        session.commit()
        print(f"Saved SourceDocument record to SQLite. Status: queued")

    # 4. Trigger background processor
    print("Running process_document_bg synchronously...")
    process_document_bg(doc_id, str(note_path), filename, DocumentSourceType.text_note)
    print("Processing finished.")

    # 5. Verify database records
    with get_session() as session:
        # Check SourceDocument
        doc = session.get(SourceDocument, doc_id)
        print("\n[ SQLite Verification ]")
        print(f"SourceDocument ID: {doc.source_document_id}")
        print(f"Filename: {doc.original_filename}")
        print(f"Ingestion Status: {doc.ingestion_status} (Expected: completed)")
        print(f"Error Message: {doc.error_message}")
        
        # Check IndexedRecords
        records = session.exec(select(IndexedRecord).where(IndexedRecord.source_document_id == doc_id)).all()
        print(f"IndexedRecord Row Count: {len(records)} (Expected: 0)")
        if len(records) > 0:
            print("[ERROR] IndexedRecord rows were written to SQLite! This violates Phase bounds.")
        else:
            print("[SUCCESS] No retrieval/index records were written to SQLite database.")
            
    # 6. Verify local metadata.json
    meta_path = DEMO_PROCESSED / doc_id / "metadata.json"
    print("\n[ File Metadata Verification ]")
    print(f"Metadata file path: {meta_path}")
    print(f"Metadata file exists: {meta_path.exists()} (Expected: True)")
    
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print("Metadata contents:")
        print(json.dumps(meta, indent=2))
        print("[SUCCESS] Local metadata.json created successfully.")
    else:
        print("[ERROR] Metadata file was not found.")
        
    # Cleanup test files from disk & db
    try:
        os.remove(note_path)
        import shutil
        shutil.rmtree(DEMO_PROCESSED / doc_id)
        with get_session() as session:
            doc = session.get(SourceDocument, doc_id)
            if doc:
                session.delete(doc)
                session.commit()
        print("\nCleanup completed.")
    except Exception as e:
        print(f"Cleanup failed: {e}")
        
    print("========================================")

if __name__ == "__main__":
    run_test()
