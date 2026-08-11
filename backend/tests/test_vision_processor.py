"""Verification: Vision processor (CLIP + LLaVA).

Tests:
  1. Full pipeline with Ollama available -> completed
  2. Partial failure when Ollama is unavailable -> partial_failed

Run:
    python -m backend.tests.test_vision_processor
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageDraw

PASS = "[PASS]"
FAIL = "[FAIL]"
TEST_DIR = Path("data/processed/_test")
SOURCE_DOC_ID = "test-vision-00000000-0000-0000-0000-000000000000"


def _create_test_image() -> Path:
    """Create a test image with shapes and text-like patterns."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    img_path = TEST_DIR / "test_vision_image.png"

    img = Image.new("RGB", (320, 240), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 150, 100], fill=(50, 100, 200), outline=(0, 0, 0))
    draw.ellipse([170, 30, 290, 110], fill=(200, 50, 50), outline=(0, 0, 0))
    draw.polygon([(160, 180), (100, 220), (220, 220)], fill=(50, 200, 100))
    img.save(img_path)
    return img_path


def test_vision_processor():
    """Run CLIP + LLaVA pipeline and check all status fields."""
    from backend.processors.vision_processor import process_image

    img_path = _create_test_image()

    print(f"\n{'='*60}")
    print(f"Test: Vision processor (CLIP + LLaVA)")
    print(f"  Image           : {img_path}")
    print(f"  Loading models (may take time on first run)...")

    result = process_image(img_path, SOURCE_DOC_ID, "test_vision_image.png")

    print(f"  CLIP status     : {result.clip_status}")
    if result.clip_embedding:
        print(f"  CLIP dimension  : {result.clip_embedding.vector_dimension}")
        print(f"  CLIP model      : {result.clip_embedding.embedding_model}")

    print(f"  LLaVA status    : {result.llava_status}")
    if result.llava_description:
        desc_preview = result.llava_description[:120].replace('\n', ' ')
        print(f"  LLaVA desc      : {desc_preview}...")
    if result.llava_description_embedding:
        print(f"  Desc embedding  : {result.llava_description_embedding.vector_dimension}-d")

    print(f"  Ingestion status: {result.ingestion_status}")
    hash_preview = (result.image_hash or 'N/A')[:40]
    print(f"  Image hash      : {hash_preview}...")

    if result.warnings:
        print(f"  Warnings:")
        for w in result.warnings:
            print(f"    - [{w.processor}] {w.message}")

    # Verify CLIP output
    clip_ok = (
        result.clip_status == "completed"
        and result.clip_embedding is not None
        and result.clip_embedding.vector_dimension == 512
    )

    # LLaVA may or may not be available
    if result.llava_status == "completed":
        llava_ok = (
            result.llava_description is not None
            and len(result.llava_description) > 10
            and result.llava_description_embedding is not None
            and result.llava_description_embedding.vector_dimension == 384
        )
        status_ok = result.ingestion_status == "completed"
        print(f"\n  LLaVA available -> expecting full completion")
    elif result.llava_status == "unavailable":
        llava_ok = (
            result.llava_description is None
            and result.llava_description_embedding is None
        )
        status_ok = result.ingestion_status == "partial_failed"
        print(f"\n  LLaVA unavailable -> expecting partial_failed (correct!)")
    else:
        llava_ok = True
        status_ok = result.ingestion_status in ("partial_failed", "failed")
        print(f"\n  LLaVA failed -> expecting partial_failed or failed")

    ok = clip_ok and llava_ok and status_ok
    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("VISION PROCESSOR VERIFICATION")
    print("CLIP: ViT-B-32 / laion2b_s34b_b79k -> 512-d")
    print("LLaVA: via Ollama (may be unavailable)")
    print("Description embedding: all-MiniLM-L6-v2 -> 384-d")
    print("=" * 60)

    results = [test_vision_processor()]

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
