"""File utility functions: hashing, safe filenames, path helpers."""

import hashlib
from pathlib import Path


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute a hex-digest hash of a file's content.

    Returns a string in the format 'algorithm:hexdigest', e.g. 'sha256:abcdef...'.
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"{algorithm}:{h.hexdigest()}"


def compute_bytes_hash(data: bytes, algorithm: str = "sha256") -> str:
    """Compute a hex-digest hash of raw bytes."""
    h = hashlib.new(algorithm)
    h.update(data)
    return f"{algorithm}:{h.hexdigest()}"


def safe_filename(filename: str) -> str:
    """Sanitize a filename, keeping only alphanumeric characters and safe punctuation."""
    keepchars = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepchars).rstrip()


# --- Supported file extensions ---

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}
SUPPORTED_BROWSER_AUDIO_EXTENSIONS = {".webm", ".ogg"}

ALL_SUPPORTED_EXTENSIONS = (
    SUPPORTED_DOCUMENT_EXTENSIONS
    | SUPPORTED_IMAGE_EXTENSIONS
    | SUPPORTED_AUDIO_EXTENSIONS
    | SUPPORTED_BROWSER_AUDIO_EXTENSIONS
)


def get_file_category(extension: str) -> str | None:
    """Return the category for a file extension, or None if unsupported.

    Categories: 'document', 'image', 'audio'.
    """
    ext = extension.lower()
    if ext in SUPPORTED_DOCUMENT_EXTENSIONS:
        return "document"
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if ext in SUPPORTED_AUDIO_EXTENSIONS or ext in SUPPORTED_BROWSER_AUDIO_EXTENSIONS:
        return "audio"
    return None
