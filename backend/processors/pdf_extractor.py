"""PDF text and image extractor using PyMuPDF (fitz).

Extracts:
- Text content page by page.
- Embedded images page by page, saved to
  ``data/processed/{source_document_id}/images/``.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from backend.processors.schemas import (
    ExtractedImageResult,
    PdfExtractionResult,
    PdfPageResult,
    ProcessorWarning,
)
from backend.utils.file_utils import compute_file_hash
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_pdf(
    file_path: str | Path,
    source_document_id: str,
    original_filename: str,
    processed_dir: str = "data/processed",
) -> PdfExtractionResult:
    """Extract text and embedded images from a PDF file.

    Parameters
    ----------
    file_path : str or Path
        Path to the PDF file.
    source_document_id : str
        UUID of the source document.
    original_filename : str
        Original user-facing filename.
    processed_dir : str
        Root directory for processed artifacts.

    Returns
    -------
    PdfExtractionResult
        Structured extraction results with per-page text and images.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    doc = fitz.open(str(file_path))
    pages: list[PdfPageResult] = []
    all_images: list[ExtractedImageResult] = []
    warnings: list[ProcessorWarning] = []

    img_dir = Path(processed_dir) / source_document_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1        # 1-based

        # --- Text ---
        page_text = page.get_text("text") or ""

        # --- Images ---
        page_images: list[ExtractedImageResult] = []
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)

                # Convert CMYK → RGB if needed
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                # Skip tiny images (likely artifacts / icons)
                if pix.width < 10 or pix.height < 10:
                    warnings.append(
                        ProcessorWarning(
                            processor="pdf_extractor",
                            message=(
                                f"Skipped tiny image ({pix.width}×{pix.height}) "
                                f"on page {page_number}, index {img_idx}"
                            ),
                        )
                    )
                    continue

                ext = "png"
                img_filename = f"page{page_number}_img{img_idx}.{ext}"
                img_path = img_dir / img_filename
                pix.save(str(img_path))

                img_hash = compute_file_hash(img_path)

                result = ExtractedImageResult(
                    source_document_id=source_document_id,
                    image_path=str(img_path),
                    image_index=img_idx,
                    page_number=page_number,
                    image_extension=ext,
                    image_hash=img_hash,
                    width=pix.width,
                    height=pix.height,
                    original_filename=original_filename,
                    source_file_path=str(file_path),
                )
                page_images.append(result)
                all_images.append(result)

            except Exception as exc:
                warnings.append(
                    ProcessorWarning(
                        processor="pdf_extractor",
                        message=(
                            f"Failed to extract image index {img_idx} "
                            f"from page {page_number}"
                        ),
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )

        pages.append(
            PdfPageResult(
                page_number=page_number,
                text=page_text,
                images=page_images,
            )
        )

    doc.close()

    logger.info(
        "PDF extracted: %s — %d pages, %d images, %d warnings",
        original_filename,
        len(pages),
        len(all_images),
        len(warnings),
    )

    return PdfExtractionResult(
        source_document_id=source_document_id,
        source_file_path=str(file_path),
        original_filename=original_filename,
        total_pages=len(pages),
        pages=pages,
        extracted_images=all_images,
        warnings=warnings,
    )
