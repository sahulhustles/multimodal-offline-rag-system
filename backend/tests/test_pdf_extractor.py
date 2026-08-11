"""Verification: PDF extractor.

Creates a test PDF with text and an embedded image using PyMuPDF,
then extracts and verifies page text and image metadata.

Run:
    python -m backend.tests.test_pdf_extractor
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fitz  # PyMuPDF

PASS = "[PASS]"
FAIL = "[FAIL]"
TEST_DIR = Path("data/processed/_test")
SOURCE_DOC_ID = "test-pdf-00000000-0000-0000-0000-000000000000"


def _create_test_pdf() -> Path:
    """Create a test PDF with text on two pages and an embedded image."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = TEST_DIR / "test_document.pdf"

    doc = fitz.open()

    # Page 1: text + image
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page 1: Introduction to Machine Learning", fontsize=16)
    page1.insert_text(
        (72, 120),
        "Machine learning is a branch of artificial intelligence that "
        "focuses on building systems that learn from data. This document "
        "tests the PDF extraction pipeline of the multimodal RAG system.",
        fontsize=11,
    )

    # Create a simple image and embed it
    from PIL import Image
    import io

    img = Image.new("RGB", (100, 80), color=(50, 100, 200))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    rect = fitz.Rect(72, 200, 172, 280)
    page1.insert_image(rect, stream=img_bytes.getvalue())

    # Page 2: more text
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page 2: Deep Learning Fundamentals", fontsize=16)
    page2.insert_text(
        (72, 120),
        "Deep learning uses neural networks with many layers. "
        "Convolutional neural networks are used for image recognition. "
        "Recurrent neural networks are used for sequence data.",
        fontsize=11,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_pdf_extraction():
    """Extract text and images from the test PDF."""
    from backend.processors.pdf_extractor import extract_pdf

    pdf_path = _create_test_pdf()
    result = extract_pdf(pdf_path, SOURCE_DOC_ID, "test_document.pdf")

    pages_ok = result.total_pages == 2
    text_ok = "Machine learning" in result.pages[0].text
    page2_ok = "Deep learning" in result.pages[1].text
    images_ok = len(result.extracted_images) >= 1

    print(f"\n{'='*60}")
    print(f"Test: PDF extraction")
    print(f"  Source document : {result.source_document_id}")
    print(f"  Total pages     : {result.total_pages}")
    print(f"  Page 1 text     : {result.pages[0].text[:80]}...")
    print(f"  Page 2 text     : {result.pages[1].text[:80]}...")
    print(f"  Extracted images: {len(result.extracted_images)}")
    for img in result.extracted_images:
        print(f"    - Page {img.page_number}: {img.image_path}")
        print(f"      Size: {img.width}×{img.height}, ext: {img.image_extension}")
        print(f"      Hash: {img.image_hash[:40]}...")
    print(f"  Warnings        : {len(result.warnings)}")
    for w in result.warnings:
        print(f"    - {w.message}")

    ok = pages_ok and text_ok and page2_ok and images_ok
    print(f"  Result          : {PASS if ok else FAIL}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("PDF EXTRACTOR VERIFICATION")
    print("=" * 60)

    results = [test_pdf_extraction()]

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
