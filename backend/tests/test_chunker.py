"""Verification: token-aware sliding-window chunker.

Tests:
  1. Text shorter than 512 tokens → single chunk
  2. Text of exactly 512 tokens → single chunk
  3. Text longer than 512 tokens → multiple chunks with overlap
  4. Overlap verification (up to 50 tokens from preceding chunk)
  5. Sentence-boundary snapping behaviour
  6. Page-aware multi-page chunking

Run:
    python -m backend.tests.test_chunker
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.processors.chunker import (
    create_chunks,
    create_chunks_from_pages,
    count_tokens,
    get_tokenizer,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
SOURCE_DOC_ID = "test-doc-00000000-0000-0000-0000-000000000000"


def test_short_text():
    """Text shorter than 512 tokens → single chunk."""
    text = "This is a short sentence. It should be one chunk."
    chunks = create_chunks(text, SOURCE_DOC_ID)
    tok_count = count_tokens(text)

    ok = len(chunks) == 1 and chunks[0].token_count == tok_count and chunks[0].token_count < 512
    print(f"\n{'='*60}")
    print(f"Test 1: Short text ({tok_count} tokens)")
    print(f"  Chunks produced : {len(chunks)}")
    print(f"  Token count     : {chunks[0].token_count}")
    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


def test_exact_512():
    """Text of exactly 512 tokens → single chunk."""
    tokenizer = get_tokenizer()
    # Build text of exactly 512 tokens
    words = []
    while True:
        words.append("hello")
        text = " ".join(words)
        if len(tokenizer.encode(text, add_special_tokens=False)) >= 512:
            break
    # Trim to exactly 512
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:512]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    actual = count_tokens(text)

    chunks = create_chunks(text, SOURCE_DOC_ID)
    ok = len(chunks) == 1 and chunks[0].token_count == actual and actual == 512

    print(f"\n{'='*60}")
    print(f"Test 2: Exactly 512 tokens")
    print(f"  Input tokens    : {actual}")
    print(f"  Chunks produced : {len(chunks)}")
    print(f"  Chunk tokens    : {chunks[0].token_count}")
    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


def test_long_text():
    """Text longer than 512 tokens → multiple chunks."""
    tokenizer = get_tokenizer()
    # Build ~1500 tokens of varied text
    sentences = [
        f"Sentence number {i} discusses topic {i % 10} in great detail."
        for i in range(200)
    ]
    text = " ".join(sentences)
    total = count_tokens(text)

    chunks = create_chunks(text, SOURCE_DOC_ID)
    all_within = all(c.token_count <= 512 for c in chunks)

    print(f"\n{'='*60}")
    print(f"Test 3: Long text ({total} tokens)")
    print(f"  Chunks produced : {len(chunks)}")
    print(f"  Token counts    : {[c.token_count for c in chunks]}")
    print(f"  All <=512 tokens: {all_within}")

    ok = len(chunks) > 1 and all_within
    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


def test_overlap():
    """Verify overlap contains up to 50 tokens from the preceding chunk."""
    tokenizer = get_tokenizer()
    sentences = [
        f"Sentence number {i} is about the topic of artificial intelligence research."
        for i in range(200)
    ]
    text = " ".join(sentences)
    chunks = create_chunks(text, SOURCE_DOC_ID)

    print(f"\n{'='*60}")
    print(f"Test 4: Overlap verification")

    ok = True
    for i in range(1, len(chunks)):
        prev_ids = tokenizer.encode(chunks[i - 1].text, add_special_tokens=False)
        curr_ids = tokenizer.encode(chunks[i].text, add_special_tokens=False)

        # Find common token suffix of prev / prefix of curr
        overlap_count = 0
        max_check = min(len(prev_ids), len(curr_ids), 60)
        for k in range(1, max_check + 1):
            if prev_ids[-k:] == curr_ids[:k]:
                overlap_count = k

        print(f"  Chunk {i-1}->{i} : overlap ~ {overlap_count} tokens (target <= 50)")
        if overlap_count > 55:  # small tolerance for sentence-boundary snapping
            ok = False

    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


def test_sentence_boundary():
    """Sentence-boundary snapping produces clean chunk endings."""
    text = (
        "The quick brown fox jumps over the lazy dog. " * 30
        + "Machine learning is transforming many industries. "
        + "Natural language processing enables computers to understand text. " * 25
    )
    chunks = create_chunks(text, SOURCE_DOC_ID)

    print(f"\n{'='*60}")
    print(f"Test 5: Sentence-boundary behaviour")
    print(f"  Total chunks: {len(chunks)}")
    ends_with_sentence = 0
    for i, c in enumerate(chunks[:-1]):  # last chunk is whatever remains
        ends_clean = c.text.rstrip().endswith((".", "!", "?"))
        if ends_clean:
            ends_with_sentence += 1
        print(f"  Chunk {i}: {c.token_count} tokens, ends with period: {ends_clean}")

    ok = ends_with_sentence > 0  # at least some should snap
    print(f"  Sentence-snapped endings: {ends_with_sentence}/{len(chunks)-1}")
    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


def test_page_aware():
    """Page-aware chunking preserves correct page numbers."""
    pages = [
        (1, "Page one content about physics and quantum mechanics. " * 80),
        (2, "Page two discusses biology and cell structure. " * 80),
        (3, "Page three covers chemistry and molecular bonds. " * 80),
    ]
    chunks = create_chunks_from_pages(pages, SOURCE_DOC_ID)

    print(f"\n{'='*60}")
    print(f"Test 6: Page-aware chunking")
    print(f"  Pages input     : {len(pages)}")
    print(f"  Chunks produced : {len(chunks)}")
    for c in chunks:
        print(f"  Chunk {c.chunk_index}: page {c.page_number}, {c.token_count} tokens")

    ok = len(chunks) > 1 and chunks[0].page_number == 1
    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("CHUNKER VERIFICATION")
    print(f"Tokenizer: {get_tokenizer().name_or_path}")
    print(f"Chunk size: 512 tokens, Overlap: 50 tokens")
    print("=" * 60)

    results = [
        test_short_text(),
        test_exact_512(),
        test_long_text(),
        test_overlap(),
        test_sentence_boundary(),
        test_page_aware(),
    ]

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
