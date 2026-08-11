"""Legacy .doc to .docx converter using LibreOffice headless mode.

Command used::

    soffice --headless --convert-to docx --outdir <dir> <input.doc>

LibreOffice must be installed on the system.  Inside Docker this is
handled by the Dockerfile (``apt-get install -y libreoffice``).
On Windows / macOS it must be installed manually.

**Fidelity warning**: Legacy .doc conversion is best-effort.
Complex formatting, macros, and embedded OLE objects may not survive
the conversion.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from backend.config import settings
from backend.processors.schemas import DocConversionResult
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


def _find_soffice() -> str | None:
    """Locate the ``soffice`` binary on the system."""
    # 1. Check configured path first
    if settings.libreoffice_path:
        cp = Path(settings.libreoffice_path)
        if cp.exists() and cp.is_file():
            return str(cp)

    # 2. Standard Linux / Docker path
    path = shutil.which("soffice")
    if path:
        return path

    # 3. Common Windows paths
    for candidate in [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]:
        if Path(candidate).exists():
            return candidate

    return None


def convert_doc_to_docx(
    doc_path: str | Path,
    output_dir: str | Path | None = None,
) -> DocConversionResult:
    """Convert a legacy ``.doc`` file to ``.docx`` via LibreOffice.

    Parameters
    ----------
    doc_path : str or Path
        Path to the ``.doc`` file.
    output_dir : str or Path, optional
        Directory for the converted ``.docx`` file.
        Defaults to the same directory as the input.

    Returns
    -------
    DocConversionResult
        ``success=True`` with ``converted_path`` on success,
        ``success=False`` with ``error_message`` on failure.
    """
    doc_path = Path(doc_path)
    if not doc_path.exists():
        return DocConversionResult(
            success=False,
            original_path=str(doc_path),
            error_message=f"Input file not found: {doc_path}",
        )

    soffice = _find_soffice()
    if soffice is None:
        return DocConversionResult(
            success=False,
            original_path=str(doc_path),
            error_message=(
                "LibreOffice (soffice) is not installed or not on PATH. "
                "Install LibreOffice to convert .doc files. "
                "On Linux: apt-get install -y libreoffice. "
                "On Windows: download from https://www.libreoffice.org."
            ),
        )

    if output_dir is None:
        output_dir = doc_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        soffice,
        "--headless",
        "--convert-to",
        "docx",
        "--outdir",
        str(output_dir),
        str(doc_path),
    ]

    try:
        logger.info("Converting .doc → .docx: %s", doc_path.name)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return DocConversionResult(
                success=False,
                original_path=str(doc_path),
                error_message=(
                    f"LibreOffice conversion failed (exit code {result.returncode}). "
                    f"stderr: {result.stderr.strip()}"
                ),
            )

        # LibreOffice writes the output as <stem>.docx in output_dir
        expected = output_dir / f"{doc_path.stem}.docx"
        if not expected.exists():
            return DocConversionResult(
                success=False,
                original_path=str(doc_path),
                error_message=(
                    f"LibreOffice ran successfully but the expected output "
                    f"file was not found: {expected}"
                ),
            )

        logger.info("DOC converted: %s → %s", doc_path.name, expected.name)
        return DocConversionResult(
            success=True,
            original_path=str(doc_path),
            converted_path=str(expected),
        )

    except subprocess.TimeoutExpired:
        return DocConversionResult(
            success=False,
            original_path=str(doc_path),
            error_message="LibreOffice conversion timed out after 120 seconds.",
        )
    except Exception as exc:
        return DocConversionResult(
            success=False,
            original_path=str(doc_path),
            error_message=f"{type(exc).__name__}: {exc}",
        )
