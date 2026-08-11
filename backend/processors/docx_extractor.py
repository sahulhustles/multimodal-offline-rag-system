"""DOCX text and image extractor using python-docx.

Extracts:
- Headings, paragraphs, and tables in reading order.
- Embedded images saved to ``data/processed/{source_document_id}/images/``.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.table import Table

from backend.processors.schemas import (
    DocxExtractionResult,
    DocxTextBlock,
    ExtractedImageResult,
    ProcessorWarning,
)
from backend.utils.file_utils import compute_file_hash, compute_bytes_hash
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


def _table_to_text(table: Table) -> str:
    """Convert a DOCX table to a readable text representation."""
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_docx(
    file_path: str | Path,
    source_document_id: str,
    original_filename: str,
    processed_dir: str = "data/processed",
) -> DocxExtractionResult:
    """Extract text blocks and embedded images from a DOCX file.

    Parameters
    ----------
    file_path : str or Path
        Path to the .docx file.
    source_document_id : str
        UUID of the source document.
    original_filename : str
        Original user-facing filename.
    processed_dir : str
        Root directory for processed artifacts.

    Returns
    -------
    DocxExtractionResult
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    doc = Document(str(file_path))
    text_blocks: list[DocxTextBlock] = []
    warnings: list[ProcessorWarning] = []
    block_index = 0

    # ---- Text: iterate body elements to preserve ordering ----
    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            # Paragraph or heading
            from docx.text.paragraph import Paragraph

            para = Paragraph(element, doc)
            text = para.text.strip()
            if not text:
                continue

            style_name = para.style.name if para.style else ""
            if style_name.startswith("Heading"):
                try:
                    level = int(style_name.replace("Heading", "").strip())
                except ValueError:
                    level = 1
                text_blocks.append(
                    DocxTextBlock(
                        block_index=block_index,
                        block_type="heading",
                        text=text,
                        heading_level=level,
                    )
                )
            else:
                text_blocks.append(
                    DocxTextBlock(
                        block_index=block_index,
                        block_type="paragraph",
                        text=text,
                    )
                )
            block_index += 1

        elif tag == "tbl":
            table = Table(element, doc)
            table_text = _table_to_text(table)
            if table_text.strip():
                text_blocks.append(
                    DocxTextBlock(
                        block_index=block_index,
                        block_type="table",
                        text=table_text,
                    )
                )
                block_index += 1

    # ---- Images: extract from document relationships ----
    extracted_images: list[ExtractedImageResult] = []
    img_dir = Path(processed_dir) / source_document_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    img_idx = 0
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            try:
                image_part = rel.target_part
                image_data: bytes = image_part.blob
                content_type: str = image_part.content_type  # e.g. "image/png"

                ext = content_type.split("/")[-1].lower()
                if ext == "jpeg":
                    ext = "jpg"

                img_filename = f"docx_img{img_idx}.{ext}"
                img_path = img_dir / img_filename

                with open(img_path, "wb") as f:
                    f.write(image_data)

                img_hash = compute_bytes_hash(image_data)

                extracted_images.append(
                    ExtractedImageResult(
                        source_document_id=source_document_id,
                        image_path=str(img_path),
                        image_index=img_idx,
                        page_number=None,        # DOCX has no page concept
                        image_extension=ext,
                        image_hash=img_hash,
                        width=None,
                        height=None,
                        original_filename=original_filename,
                        source_file_path=str(file_path),
                    )
                )
                img_idx += 1
            except Exception as exc:
                warnings.append(
                    ProcessorWarning(
                        processor="docx_extractor",
                        message=f"Failed to extract image {img_idx}",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )

    # ---- Full text for chunking ----
    full_text = "\n\n".join(b.text for b in text_blocks)

    logger.info(
        "DOCX extracted: %s — %d text blocks, %d images, %d warnings",
        original_filename,
        len(text_blocks),
        len(extracted_images),
        len(warnings),
    )

    return DocxExtractionResult(
        source_document_id=source_document_id,
        source_file_path=str(file_path),
        original_filename=original_filename,
        text_blocks=text_blocks,
        full_text=full_text,
        extracted_images=extracted_images,
        warnings=warnings,
    )
