# Ingest & Query Console Run Book

Follow these steps to initialize and start the offline Multimodal RAG demonstrator.

---

## 1. Local Infrastructure Services

### Qdrant Vector DB (Docker)
Start the local Qdrant container on port `6333`:
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Ollama Inference Engine
1. Start the Ollama Desktop application or service.
2. Download and verify the required vision-language model:
```bash
ollama pull llava
```
3. Test that the Ollama service is reachable on `http://localhost:11434/`.

---

## 2. Backend Setup & Run

### Environment Setup
Create a virtual environment (Python 3.10 to 3.12 recommended) and install dependencies:
```bash
# From workspace root (c:\Users\bsahu\Desktop\first-review)
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r backend/requirements.txt
```

### Run the FastAPI Server
Start Uvicorn with automatic reload on port `8000`:
```bash
# Run from workspace root (venv active)
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 3. Frontend Setup & Run

### Environment Setup
Install Node modules inside the `frontend` folder:
```bash
# Navigate to frontend folder
cd frontend
npm install
```

### Run the Vite Dev Server
Start the development server (usually runs on `http://localhost:5173/`):
```bash
# Run from c:\Users\bsahu\Desktop\first-review\frontend
npm run dev
```

### Compile Production Build
Verify TypeScript and Vite compilation:
```bash
# Run from c:\Users\bsahu\Desktop\first-review\frontend
npm run build
```

---

## 4. Diagnostics & Verification

### Dependency Checker
Audit active runtime dependencies (FFMPEG, Ollama, LibreOffice, Whisper):
```bash
# Run from workspace root (venv active)
python -m backend.scripts.check_runtime_dependencies
```

### Whisper Transcription test
Verify Whisper segments and speech-to-text pipeline performance:
```bash
# Run from workspace root (venv active)
python -m backend.tests.test_audio_processor
```
