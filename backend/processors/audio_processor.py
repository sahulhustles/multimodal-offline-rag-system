"""Audio transcription using faster-whisper.

Model       : large-v3
Compute type: int8
Device      : cpu (configurable)

Produces timestamped transcript segments.  Each segment becomes an
independent text chunk for downstream embedding and indexing.
"""

from __future__ import annotations

from pathlib import Path

from backend.config import settings
from backend.processors.schemas import (
    AudioSegmentResult,
    AudioTranscriptionResult,
    ProcessorWarning,
)
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton model
# ---------------------------------------------------------------------------

_model = None


def get_whisper_model():
    """Return the lazily-loaded faster-whisper model."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper model: %s (compute_type=%s, device=%s) …",
            settings.whisper_model_size,
            settings.whisper_compute_type,
            settings.whisper_device,
        )
        _model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("faster-whisper model loaded")
        try:
            from backend.utils import dependency_checker
            dependency_checker._whisper_cached_status = {
                "model_name": settings.whisper_model_size,
                "compute_type": settings.whisper_compute_type,
                "device": settings.whisper_device,
                "dependency_present": True,
                "operational": True,
                "load_test_status": "passed",
                "available": True,
                "error_message": None
            }
        except Exception:
            pass
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe_audio(
    wav_path: str | Path,
    source_document_id: str,
    original_filename: str,
) -> AudioTranscriptionResult:
    """Transcribe a normalised WAV file and return timestamped segments.

    Parameters
    ----------
    wav_path : str or Path
        Path to a 16 kHz mono WAV file (output of audio_converter).
    source_document_id : str
        UUID of the source document.
    original_filename : str
        Original user-facing filename.

    Returns
    -------
    AudioTranscriptionResult
        Contains a list of ``AudioSegmentResult`` objects, one per
        detected speech segment.

    Raises
    ------
    FileNotFoundError
        If *wav_path* does not exist.
    RuntimeError
        If the faster-whisper model fails to load.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    model = get_whisper_model()
    model_name = f"faster-whisper/{settings.whisper_model_size}"
    warnings: list[ProcessorWarning] = []

    logger.info("Transcribing: %s", wav_path.name)

    segments_iter, info = model.transcribe(
        str(wav_path),
        beam_size=5,
        vad_filter=True,
    )

    detected_language = info.language
    total_duration = info.duration

    segments: list[AudioSegmentResult] = []
    for idx, seg in enumerate(segments_iter):
        text = seg.text.strip()
        if not text:
            warnings.append(
                ProcessorWarning(
                    processor="audio_processor",
                    message=f"Empty segment at index {idx} ({seg.start:.1f}s–{seg.end:.1f}s)",
                )
            )
            continue

        segments.append(
            AudioSegmentResult(
                source_document_id=source_document_id,
                segment_index=len(segments),
                start_seconds=round(seg.start, 3),
                end_seconds=round(seg.end, 3),
                transcript_text=text,
                language=detected_language,
                model_name=model_name,
            )
        )

    logger.info(
        "Transcription complete: %s — %d segments, language=%s, duration=%.1fs",
        original_filename,
        len(segments),
        detected_language,
        total_duration or 0,
    )

    return AudioTranscriptionResult(
        source_document_id=source_document_id,
        source_file_path=str(wav_path),
        original_filename=original_filename,
        language=detected_language,
        model_name=model_name,
        segments=segments,
        total_duration_seconds=total_duration,
        warnings=warnings,
    )
