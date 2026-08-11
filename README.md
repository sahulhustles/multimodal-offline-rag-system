# Fully Offline Multimodal Retrieval-Augmented Generation System by skct students
A fully offline multimodal RAG system that ingests PDFs, DOC/DOCX, images, audio, and text notes — processes them through specialized pipelines (text chunking, CLIP vision encoding, Whisper transcription) — generates embeddings — and indexes everything into a local Qdrant vector database with cross-modal linking.

> **Current Phase**: Phase 1 — Project Scaffold & Infrastructure  
> Retrieval, reranking, and LLM answer generation are planned for future phases.

---

## Architecture Overview

```
Ingest → Process → Embed → Index → [Future: Retrieve → Rerank → Generate]
```

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Vector DB | Qdrant (Docker, local) |
| Text Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384d) |
| Image Embeddings | open_clip ViT-B-32 (512d) |
| Image Descriptions | LLaVA via Ollama |
| Audio Transcription | faster-whisper large-v3 (int8) |
| Database | SQLite via SQLModel |
| Frontend | React + Vite + TypeScript + Tailwind CSS *(Phase 5)* |

---

## Prerequisites

- **Python 3.11+** — [python.org](https://www.python.org/downloads/)
- **Docker Desktop** — [docker.com](https://www.docker.com/products/docker-desktop/)
- **Ollama** *(optional for Phase 1, required from Phase 2)* — [ollama.com](https://ollama.com)

---

## Phase 1 Setup Instructions

### 1. Clone and configure

```bash
cd first-review
copy .env.example .env
```

### 2. Start Qdrant

```bash
docker compose up -d qdrant
```

Wait for the health check to pass:

```bash
docker compose ps
# rag-qdrant should show "healthy"
```

### 3. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Start the backend

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Verify everything is running

**Health check:**

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "mock_mode": false,
  "qdrant": {
    "connected": true,
    "host": "localhost",
    "port": 6333
  },
  "ollama": {
    "connected": false,
    "model_available": false
  },
  "models": {
    "sentence_transformer": "sentence-transformers/all-MiniLM-L6-v2",
    "clip": "ViT-B-32/laion2b_s34b_b79k",
    "whisper": "large-v3 (int8)"
  }
}
```

**Collection stats:**

```bash
curl http://localhost:8000/api/v1/stats
```

Expected response:

```json
{
  "collection": {
    "collection_name": "multimodal_rag",
    "points_count": 0,
    "vectors_count": 0,
    "indexed_vectors_count": 0,
    "status": "green",
    "vectors_config": {
      "text": { "size": 384, "distance": "Cosine" },
      "image": { "size": 512, "distance": "Cosine" }
    }
  },
  "breakdown": {
    "by_source_type": {},
    "by_modality": {},
    "by_ingestion_status": {}
  }
}
```

**Qdrant dashboard:**

Open [http://localhost:6333/dashboard](http://localhost:6333/dashboard) in your browser to see the `multimodal_rag` collection.

---

## Project Structure (Phase 1)

```
first-review/
├── docker-compose.yml          # Qdrant + backend services
├── .env.example                # Environment config template
├── README.md                   # This file
│
├── backend/
│   ├── Dockerfile              # Python 3.11, ffmpeg, LibreOffice
│   ├── requirements.txt        # Python dependencies
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + lifespan
│   ├── config.py               # Pydantic Settings
│   │
│   ├── api/
│   │   ├── router.py           # API router aggregation
│   │   ├── endpoints/
│   │   │   └── health.py       # GET /health, GET /stats
│   │   └── schemas/
│   │       └── health.py       # Response models
│   │
│   ├── core/
│   │   └── exceptions.py       # Custom exception hierarchy
│   │
│   ├── db/
│   │   ├── database.py         # SQLite engine + init
│   │   └── models.py           # IngestionJob, ProcessingStep,
│   │                           # SourceDocument, IndexedRecord
│   │
│   ├── indexing/
│   │   └── qdrant_manager.py   # Collection lifecycle + stats
│   │
│   ├── processors/             # (Phase 2)
│   ├── embeddings/             # (Phase 2)
│   └── utils/
│       ├── logging_config.py   # Structured logging
│       └── file_utils.py       # Hashing, safe filenames
│
└── data/                       # Created at runtime
    ├── uploads/
    ├── processed/
    └── app.db
```

---

## Qdrant Collection Schema

**Collection name:** `multimodal_rag`

| Named Vector | Dimensions | Distance | Used by |
|-------------|-----------|----------|---------|
| `text` | 384 | Cosine | Text chunks, image descriptions, audio segments |
| `image` | 512 | Cosine | Image CLIP embeddings |

---

## API Endpoints (Phase 1)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Backend + Qdrant + Ollama health check |
| `GET` | `/api/v1/stats` | Collection statistics and breakdowns |

---

## Phase 1 & 2 Demonstrator UI (Frontend)

To assist with faculty review, a standalone React/Vite demonstrator UI is included.

### Starting the Demo UI

1. **Start the Backend APIs**:
   ```bash
   python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the Frontend UI**:
   In a new terminal window:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Access the UI**:
   Open `http://localhost:5173` in your browser.

### Teacher Demo Workflow

1. Navigate to **System Status** to verify Qdrant, Ollama, LibreOffice, and ffmpeg are ready.
2. Open the **Processor Lab**:
   - **Text Chunking**: Verify the 512/50 overlapping window.
   - **Text Embedding**: Generate a 384-dimensional text vector locally.
   - **Image Vision**: Upload an image to see CLIP extraction and LLaVA visual description.
   - **PDF/DOCX Extraction**: Upload documents and verify images and text are properly isolated.
   - **Audio Processing**: Upload audio to see it transcribed via Whisper.
3. Review **System Architecture** and **About** pages for a high-level overview.

**Important Note**: The Demonstrator UI operates on a strictly "No Side Effects" principle. All results are processed locally and stored in browser memory or the `data/demo` directory. **No Qdrant points are created during this demo (deferred to Phase 3).**

### Demo File Cleanup
You can clear temporary demo uploads and extraction artifacts by running:
```bash
python scripts/cleanup_demo.py
```

---

## Future Phases

- **Phase 3**: Indexing pipeline with cross-modal linking
- **Phase 4**: Ingestion API + background job processing
- **Phase 5**: Semantic Retrieval
- **Phase 6**: Query Processing, LLM Generation & Chat UI

---

## License

Final-year CSE project. Not for redistribution.
