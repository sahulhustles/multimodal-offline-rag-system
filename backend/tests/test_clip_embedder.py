"""Verification: CLIP image embedder.

Tests:
  1. Generate a test image with PIL, embed it → 512-d vector
  2. L2 normalisation verified
  3. Output dimension assertion

Run:
    python -m backend.tests.test_clip_embedder
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PASS = "[PASS]"
FAIL = "[FAIL]"
TEST_DIR = Path("data/processed/_test")


def _create_test_image() -> Path:
    """Create a simple test PNG image using PIL."""
    from PIL import Image, ImageDraw

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    img_path = TEST_DIR / "test_image.png"

    img = Image.new("RGB", (256, 256), color=(30, 60, 120))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 200], fill=(200, 100, 50))
    draw.ellipse([80, 80, 170, 170], fill=(50, 200, 100))
    img.save(img_path)
    return img_path


def test_clip_embedding():
    """Embed a test image → 512-d L2-normalised vector."""
    from backend.embeddings.clip_embedder import get_clip_embedder

    img_path = _create_test_image()
    embedder = get_clip_embedder()
    result = embedder.embed_image(img_path)

    ok = result.vector_dimension == 512 and len(result.vector) == 512

    norm = float(np.linalg.norm(result.vector))
    norm_ok = abs(norm - 1.0) < 1e-4

    vec = np.array(result.vector)

    print(f"\n{'='*60}")
    print(f"Test 1: CLIP image embedding")
    print(f"  Image           : {img_path}")
    print(f"  Dimension       : {result.vector_dimension}")
    print(f"  Vector shape    : {vec.shape}")
    print(f"  L2 norm         : {norm:.6f}")
    print(f"  Model           : {result.embedding_model}")
    print(f"  Preprocessing   : {result.preprocessing}")
    print(f"  Result          : {PASS if ok and norm_ok else FAIL}")
    return ok and norm_ok


def test_output_shape():
    """Output vector shape must be exactly (512,)."""
    from backend.embeddings.clip_embedder import get_clip_embedder

    img_path = _create_test_image()
    embedder = get_clip_embedder()
    result = embedder.embed_image(img_path)
    vec = np.array(result.vector)

    ok = vec.shape == (512,)
    print(f"\n{'='*60}")
    print(f"Test 2: Output shape")
    print(f"  Shape           : {vec.shape}")
    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("CLIP EMBEDDER VERIFICATION")
    print("Model: ViT-B-32 / laion2b_s34b_b79k")
    print("Expected output: 512 dimensions, L2-normalised")
    print("=" * 60)

    results = [
        test_clip_embedding(),
        test_output_shape(),
    ]

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
