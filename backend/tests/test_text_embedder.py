"""Verification: SentenceTransformer text embedder.

Tests:
  1. Short text → 384-d vector, 1 span
  2. Long text (>256 tokens) → 384-d vector, span-split + mean-pool
  3. L2 normalisation verified
  4. Output dimension assertion

Run:
    python -m backend.tests.test_text_embedder
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.embeddings.text_embedder import get_text_embedder

PASS = "[PASS]"
FAIL = "[FAIL]"


def test_short_text():
    """Short text fits in one span."""
    embedder = get_text_embedder()
    result = embedder.embed("Hello world, this is a test sentence.")

    ok = (
        result.vector_dimension == 384
        and result.embedding_span_count == 1
        and len(result.vector) == 384
    )

    norm = float(np.linalg.norm(result.vector))
    norm_ok = abs(norm - 1.0) < 1e-4

    print(f"\n{'='*60}")
    print(f"Test 1: Short text embedding")
    print(f"  Dimension       : {result.vector_dimension}")
    print(f"  Span count      : {result.embedding_span_count}")
    print(f"  L2 norm         : {norm:.6f}")
    print(f"  Model           : {result.embedding_model}")
    print(f"  Result          : {PASS if ok and norm_ok else FAIL}")
    return ok and norm_ok


def test_long_text_span_split():
    """Long text (>256 tokens) requires span-split + mean-pool."""
    embedder = get_text_embedder()
    # Build text of ~512 tokens
    text = "Artificial intelligence and machine learning are transforming industries. " * 50
    result = embedder.embed(text)

    ok = (
        result.vector_dimension == 384
        and result.embedding_span_count > 1
        and len(result.vector) == 384
    )

    norm = float(np.linalg.norm(result.vector))
    norm_ok = abs(norm - 1.0) < 1e-4

    print(f"\n{'='*60}")
    print(f"Test 2: Long text span-split")
    print(f"  Token count     : {result.token_count}")
    print(f"  Dimension       : {result.vector_dimension}")
    print(f"  Span count      : {result.embedding_span_count}")
    print(f"  L2 norm         : {norm:.6f}")
    print(f"  Result          : {PASS if ok and norm_ok else FAIL}")
    return ok and norm_ok


def test_output_shape():
    """Output vector shape must be exactly (384,)."""
    embedder = get_text_embedder()
    result = embedder.embed("Testing output shape verification.")
    vec = np.array(result.vector)

    ok = vec.shape == (384,)
    print(f"\n{'='*60}")
    print(f"Test 3: Output shape")
    print(f"  Shape           : {vec.shape}")
    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("TEXT EMBEDDER VERIFICATION")
    print("Model: sentence-transformers/all-MiniLM-L6-v2")
    print("Expected output: 384 dimensions, L2-normalised")
    print("=" * 60)

    results = [
        test_short_text(),
        test_long_text_span_split(),
        test_output_shape(),
    ]

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
