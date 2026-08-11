# Revised Architecture Plan — Mandatory Corrections Applied

> [!IMPORTANT]
> This document contains **only the revised sections** per your correction request. All other architecture decisions (offline operation, Qdrant named vectors config, embedding models, chunking 512/50, tech stack, Phases 1–2–3 processor/embedder implementation) remain **unchanged** from the original plan.

---

## 1. Revised Data-Flow Table

| Input | Processing Pipeline | Qdrant Points Created | Named Vectors Per Point |
|-------|--------------------|-----------------------|------------------------|
| **PDF** | PyMuPDF → extract text + images per page → text goes to chunker (512/50) → SentenceTransformer → indexed. Images → CLIP + LLaVA → **two linked points per image**. Cross-link text ↔ image records per page. | Text: 1 point per chunk. Per image: **2 points** (image record + description record). | Text chunk: `text` (384d). Image record: `image` (512d) only. Description record: `text` (384d) only. |
| **DOC** | LibreOffice `.doc→.docx` → python-docx → chunker → SentenceTransformer → indexed | 1 point per chunk | `text` (384d) |
| **DOCX** | python-docx → chunker → SentenceTransformer → indexed | 1 point per chunk | `text` (384d) |
| **Image** (PNG/JPG/JPEG/WEBP) | CLIP → 512d image embedding → **Point A**. LLaVA (via Ollama) → description → SentenceTransformer → 384d → **Point B**. Points A and B linked bidirectionally. | **2 points** (image record + description record) | Point A: `image` (512d) only. Point B: `text` (384d) only. |
| **Screenshot** | Same as Image. `source_type` = `screenshot` / `screenshot_description`. | **2 points** | Same as Image. |
| **Audio** (WAV/MP3/M4A + browser WebM/OGG) | ffmpeg normalize → 16kHz mono WAV → faster-whisper (large-v3, int8) → timestamped segments → each segment → SentenceTransformer. Neighbouring segments cross-linked. | 1 point per segment | `text` (384d) |
| **Text Note** | Direct text → chunker (512/50) → SentenceTransformer → indexed | 1 point per chunk | `text` (384d) |

### LLaVA Failure Scenario (any image type)

| Condition | Points Created | Status |
|-----------|---------------|--------|
| CLIP ✅ + LLaVA ✅ | 2 points (image + description), bidirectionally linked | `completed` |
| CLIP ✅ + LLaVA ❌ | **1 point only** (image record with CLIP vector) | `partial_failed` |
| Retry succeeds | Description point created, both points linked, status → `completed` | `completed` |

> [!CAUTION]
> When LLaVA fails, do **not** create a fake/placeholder description record. Only the image record is indexed. The retry endpoint creates the missing description point after LLaVA becomes available and updates both points' `linked_chunk_ids` and `ingestion_status`.

---

## 2. Revised Qdrant Point Examples — Two Linked Points Per Image

### A. Standalone Uploaded Image — CLIP Record (Point A)

```
Point ID: uuid5("image-clip-architecture_diagram.png")

Named Vectors:
  "image": [512-dim float array]         ← CLIP embedding of image pixels
  "text":  (ABSENT — not stored on this point)

Payload:
{
  "document_id": "q1r2s3t4-...",
  "chunk_id": "clip-uuid-001",
  "source_file": "data/uploads/architecture_diagram.png",
  "original_filename": "architecture_diagram.png",
  "source_type": "image",
  "modality": "image",
  "parent_document_id": null,
  "linked_chunk_ids": ["desc-uuid-001"],        ← paired description record
  "page_number": null,
  "timestamp_start": null,
  "timestamp_end": null,
  "chunk_index": 0,
  "extracted_text": null,
  "transcript_text": null,
  "llava_description": null,
  "llava_status": null,
  "created_at": "2026-07-13T22:00:00Z",
  "file_hash": "sha256:abc123...",
  "ingestion_status": "completed",
  "embedding_span_count": null,
  "embedding_model": null,
  "mock_mode": false
}
```

### B. Standalone Uploaded Image — LLaVA Description Record (Point B)

```
Point ID: uuid5("image-desc-architecture_diagram.png")

Named Vectors:
  "text":  [384-dim float array]         ← SentenceTransformer(LLaVA description)
  "image": (ABSENT — not stored on this point)

Payload:
{
  "document_id": "q1r2s3t4-...",
  "chunk_id": "desc-uuid-001",
  "source_file": "data/uploads/architecture_diagram.png",
  "original_filename": "architecture_diagram.png",
  "source_type": "image_description",
  "modality": "text",
  "parent_document_id": null,
  "linked_chunk_ids": ["clip-uuid-001"],        ← paired image record
  "page_number": null,
  "timestamp_start": null,
  "timestamp_end": null,
  "chunk_index": 0,
  "extracted_text": null,
  "transcript_text": null,
  "llava_description": "A system architecture diagram showing microservices connected via an API gateway...",
  "llava_status": "completed",
  "created_at": "2026-07-13T22:00:05Z",
  "file_hash": "sha256:abc123...",
  "ingestion_status": "completed",
  "embedding_span_count": 1,
  "embedding_model": "all-MiniLM-L6-v2",
  "mock_mode": false
}
```

---

### C. PDF Extracted Image — CLIP Record (Point A)

```
Point ID: uuid5("pdf-image-clip-report.pdf-page5-img0")

Named Vectors:
  "image": [512-dim float array]         ← CLIP embedding
  "text":  (ABSENT)

Payload:
{
  "document_id": "i1j2k3l4-...",
  "chunk_id": "pdf-clip-uuid-005",
  "source_file": "data/processed/report_page5_img0.png",
  "original_filename": "report.pdf",
  "source_type": "pdf_extracted_image",
  "modality": "image",
  "parent_document_id": "a1b2c3d4-...",          ← parent PDF's document_id
  "linked_chunk_ids": [
    "pdf-desc-uuid-005",                          ← paired description record
    "text-chunk-uuid-3",                          ← text chunks from same page
    "text-chunk-uuid-4"
  ],
  "page_number": 5,
  "chunk_index": 0,
  "extracted_text": null,
  "llava_description": null,
  "llava_status": null,
  "created_at": "2026-07-13T22:00:15Z",
  "file_hash": "sha256:def456...",
  "ingestion_status": "completed",
  "mock_mode": false
}
```

### D. PDF Extracted Image — LLaVA Description Record (Point B)

```
Point ID: uuid5("pdf-image-desc-report.pdf-page5-img0")

Named Vectors:
  "text":  [384-dim float array]         ← SentenceTransformer(LLaVA description)
  "image": (ABSENT)

Payload:
{
  "document_id": "i1j2k3l4-...",
  "chunk_id": "pdf-desc-uuid-005",
  "source_file": "data/processed/report_page5_img0.png",
  "original_filename": "report.pdf",
  "source_type": "pdf_image_description",
  "modality": "text",
  "parent_document_id": "a1b2c3d4-...",          ← parent PDF's document_id
  "linked_chunk_ids": [
    "pdf-clip-uuid-005",                          ← paired image record
    "text-chunk-uuid-3",                          ← text chunks from same page
    "text-chunk-uuid-4"
  ],
  "page_number": 5,
  "chunk_index": 0,
  "llava_description": "A bar chart showing quarterly revenue growth from Q1-Q4 2025...",
  "llava_status": "completed",
  "created_at": "2026-07-13T22:00:20Z",
  "file_hash": "sha256:def456...",
  "ingestion_status": "completed",
  "embedding_span_count": 1,
  "embedding_model": "all-MiniLM-L6-v2",
  "mock_mode": false
}
```

---

### E. PDF Extracted Image — LLaVA FAILED (Only Point A Created)

```
Point ID: uuid5("pdf-image-clip-report.pdf-page8-img1")

Named Vectors:
  "image": [512-dim float array]         ← CLIP still works
  "text":  (ABSENT)

Payload:
{
  "document_id": "x1y2z3w4-...",
  "chunk_id": "pdf-clip-uuid-008",
  "source_file": "data/processed/report_page8_img1.png",
  "original_filename": "report.pdf",
  "source_type": "pdf_extracted_image",
  "modality": "image",
  "parent_document_id": "a1b2c3d4-...",
  "linked_chunk_ids": [],                         ← NO description record exists
  "page_number": 8,
  "chunk_index": 0,
  "llava_description": null,
  "llava_status": "unavailable",
  "ingestion_status": "partial_failed",           ← clearly marked
  "created_at": "2026-07-13T22:00:30Z",
  "file_hash": "sha256:ghi789...",
  "mock_mode": false
}

NO description point created. No fake data.
Retry endpoint will:
  1. Call LLaVA for description
  2. Create new description point (Point B)
  3. Update this point's linked_chunk_ids to include Point B's chunk_id
  4. Update ingestion_status → "completed"
```

---

### F. Screenshot — Two Linked Points

```
Point A (CLIP Record):
  source_type: "screenshot"
  modality: "image"
  Named Vectors: "image" = [512-dim], "text" = (ABSENT)
  linked_chunk_ids: ["desc-uuid-screenshot-001"]

Point B (Description Record):
  source_type: "screenshot_description"
  modality: "text"
  Named Vectors: "text" = [384-dim], "image" = (ABSENT)
  linked_chunk_ids: ["clip-uuid-screenshot-001"]
  llava_description: "A screenshot of a terminal window showing Docker logs..."
```

---

### Complete `source_type` Enumeration (Updated)

| `source_type` value | When used | `modality` |
|---------------------|-----------|------------|
| `pdf` | PDF text chunk | `text` |
| `doc` | DOC text chunk (after conversion) | `text` |
| `docx` | DOCX text chunk | `text` |
| `image` | Standalone uploaded image — CLIP record | `image` |
| `image_description` | Standalone uploaded image — LLaVA description record | `text` |
| `screenshot` | Screenshot — CLIP record | `image` |
| `screenshot_description` | Screenshot — LLaVA description record | `text` |
| `pdf_extracted_image` | Image extracted from PDF — CLIP record | `image` |
| `pdf_image_description` | Image extracted from PDF — LLaVA description record | `text` |
| `audio` | Audio transcript segment | `audio` |
| `text_note` | User text note chunk | `text` |

---

## 3. Revised Cross-Modal Linking Table

| Scenario | Point's `linked_chunk_ids` contains |
|----------|--------------------------------------|
| **PDF text chunk** on page N | All **image record** chunk_ids extracted from page N + optionally all **image-description record** chunk_ids from page N |
| **PDF extracted image record** (CLIP) on page N | Its paired **image-description record** chunk_id (if LLaVA succeeded) + all **text chunk** IDs from page N |
| **PDF image-description record** on page N | Its paired **image record** chunk_id + all **text chunk** IDs from page N |
| **Standalone image record** (CLIP) | Its paired **image-description record** chunk_id (if LLaVA succeeded) |
| **Standalone image-description record** | Its paired **image record** chunk_id |
| **Screenshot image record** | Its paired **screenshot-description record** chunk_id (if LLaVA succeeded) |
| **Screenshot-description record** | Its paired **screenshot image record** chunk_id |
| **Audio segment** at index K | Segment K-1 and segment K+1 chunk_ids (where available) |
| **Text note chunk** at index K | Adjacent chunk K-1 and K+1 from the same note (where available) |
| **LLaVA failed** image record | `linked_chunk_ids = []` (empty — no description record exists) |

### Linking Invariants

1. Every image record links to **exactly one** description record when LLaVA succeeds, **zero** when it fails.
2. Every description record links to **exactly one** image record — always.
3. PDF-context links (text ↔ image, text ↔ description) are **bidirectional**: if text chunk T links to image record I, then I also links to T.
4. Audio neighbour links are **bidirectional**: if segment K links to K+1, then K+1 also links to K.
5. Retry creates the missing description point **and** updates the image record's `linked_chunk_ids` atomically.

---

## 4. Updated Frontend Folder Structure

```
frontend/src/
├── main.tsx
├── App.tsx                              # Router with all 6 pages
├── index.css                            # Tailwind + dark theme
│
├── api/
│   └── client.ts                        # Axios/fetch wrapper
│
├── components/
│   ├── Layout.tsx                       # Sidebar + top bar shell
│   ├── FileUpload.tsx                   # Drag-and-drop + browse
│   ├── AudioRecorder.tsx                # MediaRecorder → webm
│   ├── TextNoteInput.tsx                # Title + textarea
│   ├── JobList.tsx                      # Filterable job list
│   ├── JobCard.tsx                      # Per-job timeline + vector status + retry
│   ├── StatsPanel.tsx                   # Collection stats
│   ├── PayloadInspector.tsx             # Vector names + dims (no raw arrays)
│   ├── MockModeBanner.tsx               # Global mock-mode warning
│   ├── DocumentTable.tsx                # [NEW] Reusable sortable/filterable doc table
│   ├── LinkedRecordChips.tsx            # [NEW] Clickable chips for linked chunk_ids
│   ├── VectorPresenceBadge.tsx          # [NEW] Shows which named vectors are present
│   └── Toast.tsx                        # Notifications
│
├── pages/
│   ├── Dashboard.tsx                    # Stats + recent jobs overview
│   ├── UploadPage.tsx                   # File upload + text note + audio
│   ├── JobsPage.tsx                     # Full job history
│   ├── DocumentsPage.tsx                # [NEW] Searchable/filterable document list
│   ├── DocumentDetailPage.tsx           # [NEW] Single-doc detail view
│   └── ArchitecturePage.tsx             # [NEW] Static system architecture diagram
│
├── hooks/
│   ├── usePolling.ts                    # Generic polling for job status
│   └── useAudioRecorder.ts             # Browser microphone hook
│
└── types/
    └── index.ts                         # Shared TypeScript interfaces
```

### New Page Descriptions

#### `DocumentsPage.tsx`
- Displays all indexed documents in a searchable, filterable table.
- Columns: original filename, source type, modality, ingestion status, vector presence (text ✓/✗, image ✓/✗), created_at.
- Filters: by `source_type`, by `modality`, by `ingestion_status`.
- Search: by filename or extracted text substring.
- Each row links to `DocumentDetailPage`.

#### `DocumentDetailPage.tsx`
- **Source preview**: image thumbnail (for image/screenshot), text snippet (for text types), audio player stub (for audio).
- **Metadata panel**: document_id, chunk_id, source_type, modality, file_hash, page_number, timestamps.
- **Chunks section**: for multi-chunk documents (PDFs, long notes), list all chunks with chunk_index.
- **Vector presence**: badges showing which named vectors (`text`, `image`) are present with their dimensions.
- **Linked records**: clickable chips for each entry in `linked_chunk_ids` — clicking navigates to that record's detail page.
- **JSON payload viewer**: collapsible raw payload view (excludes raw embedding arrays).
- **Retry action**: if `ingestion_status === "partial_failed"`, show a "Retry LLaVA Description" button.

#### `ArchitecturePage.tsx`
- Static visual representation of the full system architecture.
- Sections rendered as styled cards or a flowchart:

```
┌─────────────────── INGEST ───────────────────┐
│  PDF / DOC    Images     Audio    User Query  │
└──────────────────┬───────────────────────────┘
                   │
┌─────────────────── PROCESS ──────────────────┐
│  Text Extractor  │ Vision Encoder │ Whisper   │
│  (PyMuPDF,       │ (CLIP +        │ (faster-  │
│   python-docx,   │  LLaVA/Ollama) │  whisper  │
│   chunker 512/50)│                │  large-v3)│
│                  │ Query Encoder              │
│                  │ (future scope)             │
└──────────────────┬───────────────────────────┘
                   │
┌─────────────────── INDEX ────────────────────┐
│        Unified Qdrant Vector Store           │
│   Collection: multimodal_rag                 │
│   Named vectors: text(384d), image(512d)     │
└──────────────────┬───────────────────────────┘
                   │
┌─── RETRIEVE (Future Scope — Not Implemented) ┐
│  Retrieved Context  │  Cross-Encoder Reranker │
└──────────────────┬───────────────────────────┘
                   │
┌─── GENERATE (Future Scope — Not Implemented) ┐
│        Local LLM / Ollama                    │
│        Answer Generation                     │
└──────────────────────────────────────────────┘
```

- The Retrieve and Generate sections use a visually distinct style (dashed borders, muted colors, "🔒 Future Scope" badge) to clearly indicate they are not implemented.
- Hovering or clicking each component shows a tooltip with the specific model/library used.

---

## 5. Updated API Endpoint List

### Ingestion Endpoints (`/api/v1/ingest`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest/upload` | Upload files (PDF, DOC, DOCX, images, audio). Multipart form. Returns `job_id`. |
| `POST` | `/ingest/text-note` | Submit text note with optional title. Returns `job_id`. |
| `POST` | `/ingest/audio-recording` | Upload browser-recorded audio blob. Returns `job_id`. |

### Job Endpoints (`/api/v1/jobs`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/jobs` | List all jobs. Supports `?status=`, `?page=`, `?per_page=`. |
| `GET` | `/jobs/{job_id}` | Detailed job status with processing timeline, per-step vector status, and document list. |

### Document Endpoints (`/api/v1/documents`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/documents` | **[NEW]** List all indexed documents. Supports `?source_type=`, `?modality=`, `?status=`, `?search=`, `?page=`, `?per_page=`. |
| `GET` | `/documents/{document_id}` | **[NEW]** Full document detail: metadata, vector presence, linked records, payload (no raw arrays). |
| `POST` | `/documents/{document_id}/retry-vision-description` | Retry LLaVA → creates missing description point, links both points, updates status. |

### Health & Stats Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Backend + Qdrant + Ollama connectivity + model availability. |
| `GET` | `/stats` | Collection stats: point count, by source_type, by modality, by ingestion_status. |

### Future Endpoints (Stub Only)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Placeholder for retrieval + RAG query. Returns `501 Not Implemented`. |

### Key Change: Retry Endpoint Behavior (Updated)

The `POST /documents/{document_id}/retry-vision-description` endpoint now:

1. Accepts the `document_id` of an image record with `ingestion_status: "partial_failed"`.
2. Calls LLaVA via Ollama to generate a description.
3. Embeds the description with SentenceTransformer (384d).
4. **Creates a new Qdrant point** (the description record) with:
   - `source_type`: `image_description` / `screenshot_description` / `pdf_image_description`
   - `modality`: `text`
   - `linked_chunk_ids`: `[original_image_record_chunk_id]`
5. **Updates the original image record point** in Qdrant:
   - Adds the new description record's chunk_id to `linked_chunk_ids`
   - Sets `ingestion_status` → `completed`
6. Returns `{ status: "completed", description_chunk_id: "..." }`.

---

## Summary of All Changes

| Area | Before (Original Plan) | After (Revised) |
|------|----------------------|-----------------|
| Image indexing | 1 point per image with both `image` + `text` named vectors | **2 linked points**: image record (`image` vector only) + description record (`text` vector only) |
| `source_type` values | 7 values | **11 values** (added `pdf_extracted_image`, `image_description`, `screenshot_description`, `pdf_image_description`) |
| LLaVA failure | 1 point with CLIP vector, text vector absent | 1 point (image record only), **no description point created**, retry creates it later |
| Retry endpoint | Updates existing point | **Creates new point** + updates existing point's links |
| Cross-modal linking | Image CLIP ↔ description in same point | Bidirectional `linked_chunk_ids` between two separate points |
| Frontend pages | 3 pages (Dashboard, Upload, Jobs) | **6 pages** (+DocumentsPage, DocumentDetailPage, ArchitecturePage) |
| Frontend components | 10 components | **13 components** (+DocumentTable, LinkedRecordChips, VectorPresenceBadge) |
| API endpoints | 9 endpoints | **11 endpoints** (+GET /documents, +GET /documents/{id}) |

> [!NOTE]
> All other architecture decisions remain unchanged: offline operation, Qdrant Docker config, collection name `multimodal_rag`, named vectors (`text`: 384d cosine, `image`: 512d cosine), chunking (512/50), embedding models, audio processing, no retrieval/reranking/LLM generation in this phase, mock mode rules, Docker Compose setup, folder structure for backend modules.

---

**Awaiting your approval to proceed with implementation.**
