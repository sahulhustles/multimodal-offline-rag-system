"""Application configuration loaded from environment variables via pydantic-settings."""

import os
from pathlib import Path

# Force HuggingFace libraries to run in offline mode and use PyTorch only (skip TensorFlow)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Multimodal RAG backend.

    All settings can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "Multimodal RAG System"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- Mock Mode ---
    enable_demo_mock_mode: bool = False

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "multimodal_rag"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llava"

    # --- Embedding Models ---
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    # --- Whisper ---
    whisper_model_size: str = "large-v3"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # --- Storage ---
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    database_url: str = "sqlite:///data/app.db"

    # --- Chunking ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # --- Indexing (Phase 3) ---
    max_images_per_document: int = 50
    qdrant_upsert_batch_size: int = 64

    # --- Retrieval (Phase 4) ---
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.0

    # --- Generation (Phase 5) ---
    # Model used for answer generation (separate from ollama_model used for image descriptions)
    generation_model: str = "llama3.2:3b"
    generation_temperature: float = 0.0
    generation_max_tokens: int = 512
    generation_timeout: float = 120.0
    # Evidence gate: minimum retrieval score to include a record in the LLM context.
    # Calibrated to 0.10 to prevent dropping relevant semantic matches.
    generation_min_score: float = 0.10
    generation_max_context_records: int = 5
    generation_max_chars_per_record: int = 2000
    generation_max_context_chars: int = 8000


    # --- Server ---
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # --- External Dependencies ---
    ffmpeg_path: str | None = None
    libreoffice_path: str | None = None

    # --- Derived Paths ---

    @property
    def upload_path(self) -> Path:
        """Return upload directory path, creating it if needed."""
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def processed_path(self) -> Path:
        """Return processed artifacts directory path, creating it if needed."""
        p = Path(self.processed_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
