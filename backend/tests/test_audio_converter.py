"""Verification: audio normaliser (ffmpeg).

Creates a test WAV file using Python's wave module,
then normalises it to 16 kHz mono via ffmpeg.

Run:
    python -m backend.tests.test_audio_converter
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
SOURCE_DOC_ID = "test-audio-00000000-0000-0000-0000-000000000000"


def _create_test_wav(sample_rate: int = 44100, duration: float = 2.0) -> Path:
    """Create a simple 440 Hz sine wave WAV at the given sample rate."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = TEST_DIR / "test_audio_44100.wav"

    num_samples = int(sample_rate * duration)
    frequency = 440.0

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(2)          # stereo (to test mono conversion)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)

        for i in range(num_samples):
            value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate))
            # Write same value to both channels
            wf.writeframes(struct.pack("<hh", value, value))

    return wav_path


def test_audio_normalisation():
    """Normalise a 44.1 kHz stereo WAV → 16 kHz mono WAV."""
    from backend.processors.audio_converter import normalize_audio

    wav_path = _create_test_wav()
    result = normalize_audio(wav_path, SOURCE_DOC_ID)

    print(f"\n{'='*60}")
    print(f"Test: Audio normalisation")
    print(f"  Input           : {result.original_path}")
    print(f"  Original format : {result.original_format}")
    print(f"  Success         : {result.success}")

    if result.success:
        print(f"  Output          : {result.normalized_path}")
        dur = result.duration_seconds
        print(f"  Duration        : {f'{dur:.1f}s' if dur else 'N/A (ffprobe not found)'}")

        # Verify output is 16 kHz mono
        with wave.open(result.normalized_path, "r") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            print(f"  Output channels : {channels} (expected 1)")
            print(f"  Output rate     : {rate} Hz (expected 16000)")

        ok = channels == 1 and rate == 16000
    else:
        print(f"  Error           : {result.error_message}")
        ok = False

    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO CONVERTER VERIFICATION")
    print("=" * 60)

    results = [test_audio_normalisation()]

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
