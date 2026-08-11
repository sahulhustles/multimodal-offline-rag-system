"""Verification: faster-whisper audio transcription.

Uses the normalised WAV from the audio converter test to run
faster-whisper large-v3 (int8) transcription.

NOTE: This test loads the large-v3 model (~3 GB download on first run).
      On a CPU-only machine, transcription may take 30+ seconds.

Run:
    python -m backend.tests.test_audio_processor
"""

import math
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PASS = "[PASS]"
FAIL = "[FAIL]"
TEST_DIR = Path("data/processed/_test")
SOURCE_DOC_ID = "test-whisper-00000000-0000-0000-0000-000000000000"


def _create_16khz_wav(duration: float = 3.0) -> Path:
    """Create a 16 kHz mono WAV (silence) for transcription test."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = TEST_DIR / "test_audio_16khz.wav"

    sample_rate = 16000
    num_samples = int(sample_rate * duration)

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write near-silence (small sine wave, Whisper should detect "no speech")
        for i in range(num_samples):
            value = int(100 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wf.writeframes(struct.pack("<h", value))

    return wav_path


def test_whisper_transcription():
    """Transcribe a test WAV and verify segment structure."""
    from backend.processors.audio_processor import transcribe_audio

    wav_path = _create_16khz_wav()

    print(f"\n{'='*60}")
    print(f"Test: Whisper transcription")
    print(f"  Input           : {wav_path}")
    print(f"  Model           : large-v3 (int8)")
    print(f"  Loading model (may take time on first run)...")

    try:
        result = transcribe_audio(wav_path, SOURCE_DOC_ID, "test_audio.wav")

        print(f"  Language        : {result.language}")
        print(f"  Duration        : {result.total_duration_seconds:.1f}s")
        print(f"  Segments        : {len(result.segments)}")
        print(f"  Model name      : {result.model_name}")

        for seg in result.segments[:5]:
            print(
                f"    [{seg.start_seconds:.1f}s – {seg.end_seconds:.1f}s] "
                f"{seg.transcript_text[:60]}"
            )

        if result.warnings:
            print(f"  Warnings:")
            for w in result.warnings:
                print(f"    - {w.message}")

        # Even silence/noise should produce a valid result object
        ok = (
            result.source_document_id == SOURCE_DOC_ID
            and result.model_name.startswith("faster-whisper/")
            and result.total_duration_seconds is not None
        )

        print(f"  Result          : {PASS if ok else FAIL}")
        return ok

    except Exception as exc:
        print(f"  ERROR           : {type(exc).__name__}: {exc}")
        print(f"  Result          : {FAIL}")
        print(f"\n  NOTE: Ensure faster-whisper is installed and the model can be downloaded.")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("WHISPER TRANSCRIPTION VERIFICATION")
    print("=" * 60)

    results = [test_whisper_transcription()]

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
