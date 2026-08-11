"""Custom exception hierarchy for the Multimodal RAG System.

All domain exceptions inherit from RAGBaseError so callers can catch
the full family or individual subclasses as needed.
"""


class RAGBaseError(Exception):
    """Base exception for all RAG system errors."""

    def __init__(self, message: str = "An error occurred in the RAG system."):
        self.message = message
        super().__init__(self.message)


class UnsupportedFormatError(RAGBaseError):
    """Raised when a file format is not supported."""

    def __init__(self, filename: str, extension: str):
        self.filename = filename
        self.extension = extension
        super().__init__(
            f"Unsupported file format '{extension}' for file '{filename}'. "
            "Supported: PDF, DOC, DOCX, PNG, JPG, JPEG, WEBP, WAV, MP3, M4A."
        )


class ProcessingError(RAGBaseError):
    """Raised when a processing step fails."""

    def __init__(self, step: str, detail: str):
        self.step = step
        super().__init__(f"Processing failed at step '{step}': {detail}")


class ModelUnavailableError(RAGBaseError):
    """Raised when a required ML model is not available locally."""

    def __init__(self, model_name: str, detail: str = ""):
        self.model_name = model_name
        msg = f"Model '{model_name}' is not available."
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


class QdrantConnectionError(RAGBaseError):
    """Raised when Qdrant is not reachable."""

    def __init__(self, detail: str = ""):
        msg = "Cannot connect to Qdrant vector database."
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


class DocumentNotFoundError(RAGBaseError):
    """Raised when a requested source document or indexed record is not found."""

    def __init__(self, identifier: str, entity_type: str = "record"):
        self.identifier = identifier
        super().__init__(f"{entity_type.capitalize()} '{identifier}' not found.")


class RetryNotAllowedError(RAGBaseError):
    """Raised when a retry is attempted on a record that is not in partial_failed state."""

    def __init__(self, chunk_id: str, current_status: str):
        self.chunk_id = chunk_id
        super().__init__(
            f"Retry not allowed for record '{chunk_id}' with status '{current_status}'. "
            "Only 'partial_failed' records can be retried."
        )
