"""Verification: .doc to .docx converter.

Tests that LibreOffice headless is available and can convert files,
or returns a clear dependency error if it is not installed.

Run:
    python -m backend.tests.test_doc_converter
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PASS = "[PASS]"
FAIL = "[FAIL]"
TEST_DIR = Path("data/processed/_test")


def test_doc_converter():
    """Test .doc -> .docx conversion or clear error message."""
    from backend.processors.doc_converter import convert_doc_to_docx, _find_soffice

    soffice = _find_soffice()

    print(f"\n{'='*60}")
    print(f"Test: DOC -> DOCX converter")
    print(f"  LibreOffice found: {soffice is not None}")
    if soffice:
        print(f"  Path: {soffice}")

    # Test with a non-existent file to verify error handling
    result = convert_doc_to_docx("nonexistent.doc", TEST_DIR)
    file_not_found_ok = not result.success and "not found" in (result.error_message or "")
    print(f"  Missing-file error: {result.error_message}")
    print(f"  Missing-file handled: {file_not_found_ok}")

    if soffice is None:
        # LibreOffice not available -- verify we get a clear error
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        dummy = TEST_DIR / "dummy.doc"
        dummy.write_bytes(b"dummy content")
        result = convert_doc_to_docx(dummy, TEST_DIR)
        clear_error = not result.success and "LibreOffice" in (result.error_message or "")
        print(f"  No-soffice error: {result.error_message}")
        print(f"  Clear dependency error: {clear_error}")
        ok = file_not_found_ok and clear_error
        print(f"  Result: {PASS if ok else FAIL}")
        print(f"\n  NOTE: LibreOffice is not installed. Install it to enable .doc conversion.")
        return ok
    else:
        ok = file_not_found_ok
        print(f"  Result: {PASS if ok else FAIL}")
        print(f"  NOTE: LibreOffice is available. Full conversion test requires a .doc file.")
        return ok


if __name__ == "__main__":
    print("=" * 60)
    print("DOC CONVERTER VERIFICATION")
    print("=" * 60)

    results = [test_doc_converter()]

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} tests passed")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
