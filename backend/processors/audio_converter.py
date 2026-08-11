"""Audio normalisation via ffmpeg.

Converts any supported audio format to:
- WAV
- 16 kHz sample rate
- Mono channel

Normalised files are saved under
``data/processed/{source_document_id}/audio/``.

Supported input formats: WAV, MP3, M4A, WebM, OGG.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from backend.processors.schemas import AudioNormalizationResult
from backend.utils.dependency_checker import get_ffmpeg_status
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


def _get_duration(file_path: str, ffprobe: str | None = None) -> float | None:
    """Get audio duration in seconds using ffprobe."""
    probe = ffprobe or shutil.which("ffprobe")
    if not probe:
        return None
    try:
        result = subprocess.run(
            [
                probe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def normalize_audio(
    audio_path: str | Path,
    source_document_id: str,
    processed_dir: str = "data/processed",
) -> AudioNormalizationResult:
    """Normalise an audio file to WAV 16 kHz mono via ffmpeg.

    Parameters
    ----------
    audio_path : str or Path
        Path to the input audio file.
    source_document_id : str
        UUID of the source document.
    processed_dir : str
        Root directory for processed artifacts.

    Returns
    -------
    AudioNormalizationResult
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        return AudioNormalizationResult(
            success=False,
            original_path=str(audio_path),
            original_format=audio_path.suffix.lstrip("."),
            error_message=f"Audio file not found: {audio_path}",
        )

    ffmpeg_status = get_ffmpeg_status()
    if not ffmpeg_status["available"]:
        return AudioNormalizationResult(
            success=False,
            original_path=str(audio_path),
            original_format=audio_path.suffix.lstrip("."),
            error_message=(
                "ffmpeg is not installed or not on PATH. "
                f"Detection details: {ffmpeg_status['error_message']}"
            ),
        )
        
    ffmpeg = ffmpeg_status["resolved_path"]

    original_format = audio_path.suffix.lstrip(".").lower()

    # Output path
    out_dir = Path(processed_dir) / source_document_id / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{audio_path.stem}_normalized.wav"

    cmd = [
        ffmpeg,
        "-y",               # overwrite
        "-i", str(audio_path),
        "-ar", "16000",      # 16 kHz
        "-ac", "1",          # mono
        "-f", "wav",
        str(out_path),
    ]

    try:
        logger.info("Normalising audio: %s → WAV 16kHz mono", audio_path.name)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            return AudioNormalizationResult(
                success=False,
                original_path=str(audio_path),
                original_format=original_format,
                error_message=(
                    f"ffmpeg conversion failed (exit {result.returncode}). "
                    f"stderr: {result.stderr[:500]}"
                ),
            )

        if not out_path.exists():
            return AudioNormalizationResult(
                success=False,
                original_path=str(audio_path),
                original_format=original_format,
                error_message=f"ffmpeg completed but output not found: {out_path}",
            )

        duration = _get_duration(str(out_path))

        logger.info(
            "Audio normalised: %s (%.1fs)",
            out_path.name,
            duration or 0,
        )

        return AudioNormalizationResult(
            success=True,
            original_path=str(audio_path),
            normalized_path=str(out_path),
            original_format=original_format,
            duration_seconds=duration,
        )

    except subprocess.TimeoutExpired:
        return AudioNormalizationResult(
            success=False,
            original_path=str(audio_path),
            original_format=original_format,
            error_message="ffmpeg normalisation timed out after 300 seconds.",
        )
    except Exception as exc:
        return AudioNormalizationResult(
            success=False,
            original_path=str(audio_path),
            original_format=original_format,
            error_message=f"{type(exc).__name__}: {exc}",
        )
