"""Citation validator for Phase 5 grounded answer generation.

Extracts ``[S#]`` citation patterns from the LLM answer, validates them
against the set of allowed source IDs from the assembled context, and
enforces the user's strict safety requirements:

- If ANY citation is outside the allowed set → generation-validation failure.
- If the answer has ZERO valid citations → generation-validation failure.
- Only a fully valid answer (all citations allowed, ≥1 citation) is grounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.generation.prompt_builder import INSUFFICIENT_CONTEXT_PHRASE
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# Regex to find all [S#] patterns in the answer
_CITATION_RE = re.compile(r"\[S(\d+)\]")

# The safe message returned on validation failure
VALIDATION_FAILURE_MESSAGE = (
    "I could not verify the generated answer against the retrieved sources. "
    "Please review the retrieved evidence."
)


@dataclass
class CitationResult:
    """Result of citation validation."""

    # The validated answer text (or the failure message)
    answer: str

    # Whether the answer is grounded (all citations valid, ≥1 citation)
    grounded: bool = False

    # Whether the LLM returned the insufficient-context phrase
    insufficient_context: bool = False

    # List of valid cited source IDs (e.g. ["S1", "S3"])
    cited_ids: list[str] = field(default_factory=list)

    # Warnings for logging / response metadata
    warnings: list[str] = field(default_factory=list)

    # Whether validation failed (invalid/missing citations)
    validation_failed: bool = False


def validate_citations(
    raw_answer: str,
    allowed_ids: set[str],
) -> CitationResult:
    """Validate citations in a generated answer against allowed source IDs.

    Parameters
    ----------
    raw_answer:
        The raw text returned by the LLM.
    allowed_ids:
        Set of allowed source IDs (e.g. ``{"S1", "S2", "S3"}``).

    Returns
    -------
    CitationResult
        Validation result with grounded status, warnings, and the
        answer text (either the original or the safe failure message).
    """
    result = CitationResult(answer=raw_answer)

    # Check for insufficient-context phrase
    normalised = raw_answer.strip().lower()
    insufficient_normalised = INSUFFICIENT_CONTEXT_PHRASE.strip().lower()

    if normalised.startswith(insufficient_normalised) or normalised == insufficient_normalised:
        result.insufficient_context = True
        result.grounded = False
        result.answer = INSUFFICIENT_CONTEXT_PHRASE
        logger.info("LLM returned insufficient-context response")
        return result

    # Extract all [S#] citations
    matches = _CITATION_RE.findall(raw_answer)
    cited_ids = {f"S{m}" for m in matches}

    if not cited_ids:
        # Zero citations in a non-empty answer → validation failure
        result.validation_failed = True
        result.grounded = False
        result.answer = VALIDATION_FAILURE_MESSAGE
        result.warnings.append(
            "Generated answer contained no [S#] citations. "
            "Answer suppressed for safety."
        )
        logger.warning(
            "Citation validation FAILED: no citations found in answer (%d chars)",
            len(raw_answer),
        )
        return result

    # Check for invalid citations
    invalid_ids = cited_ids - allowed_ids
    valid_ids = cited_ids & allowed_ids

    if invalid_ids:
        # Any invalid citation → validation failure, do NOT expose the answer
        result.validation_failed = True
        result.grounded = False
        result.answer = VALIDATION_FAILURE_MESSAGE
        result.cited_ids = sorted(valid_ids)
        result.warnings.append(
            f"Generated answer cited invalid source IDs: {sorted(invalid_ids)}. "
            "Answer suppressed for safety."
        )
        logger.warning(
            "Citation validation FAILED: invalid IDs %s (allowed: %s)",
            sorted(invalid_ids),
            sorted(allowed_ids),
        )
        return result

    # All citations are valid and at least one exists → grounded
    result.grounded = True
    result.cited_ids = sorted(valid_ids)
    result.answer = raw_answer

    logger.info(
        "Citation validation PASSED: cited %s (all in allowed set %s)",
        sorted(valid_ids),
        sorted(allowed_ids),
    )

    return result
