"""Intent router and contextual query resolver for conversational interaction.

Classifies incoming user messages into:
- CONVERSATIONAL: greetings, small talk, meta-questions about the assistant or system
- DOCUMENT_QUERY: questions seeking factual information from documents

Also resolves coreferences/pronouns in follow-up queries using conversation history.
"""

from __future__ import annotations

import enum
import re
from typing import Sequence

from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class QueryIntent(str, enum.Enum):
    """Classification of user query intent."""

    CONVERSATIONAL = "conversational"
    DOCUMENT_QUERY = "document_query"


# Exact or start-with conversational phrases
_GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|howdy|hola|yo)[\s!.,?]*$",
    r"^(good\s+(morning|afternoon|evening|day|night))[\s!.,?]*$",
    r"^(how\s+are\s+you|how\s+is\s+it\s+going|how's\s+it\s+going|what's\s+up)[\s!.,?]*$",
    r"^(who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name|tell\s+me\s+about\s+yourself)[\s!.,?]*$",
    r"^(what\s+can\s+you\s+do|how\s+can\s+you\s+help|what\s+are\s+your\s+capabilities|help(\s+me)?)[\s!.,?]*$",
    r"^(thank\s+you(\s+(so|very)\s+much)?|thanks(\s+a\s+lot)?|thx|cool|awesome|great|ok|okay)[\s!.,?]*$",
    r"^(bye|goodbye|see\s+you|cya)[\s!.,?]*$",
]

_GREETING_REGEXES = [re.compile(p, re.IGNORECASE) for p in _GREETING_PATTERNS]

# Pronouns or follow-up markers that indicate a query depends on previous turns
_FOLLOWUP_PRONOUNS = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|it|its|that|this|the same|also|more details|elaborate)\b",
    re.IGNORECASE,
)


_LEADING_GREETING_RE = re.compile(
    r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day))\b[,\s!]*",
    re.IGNORECASE,
)

_CONVERSATIONAL_INTENT_RE = re.compile(
    r"\b(who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name|tell\s+me\s+about\s+yourself|"
    r"what\s+can\s+you\s+do|how\s+can\s+you\s+help|what\s+are\s+your\s+capabilities|"
    r"how\s+are\s+you|how's\s+it\s+going|how\s+is\s+it\s+going|what's\s+up)\b",
    re.IGNORECASE,
)


def classify_intent(query: str) -> QueryIntent:
    """Classify whether a query is conversational or a document question.

    Parameters
    ----------
    query:
        Raw user input.

    Returns
    -------
    QueryIntent
        CONVERSATIONAL or DOCUMENT_QUERY.
    """
    cleaned = query.strip()

    # Empty or very short conversational tokens
    if not cleaned:
        return QueryIntent.CONVERSATIONAL

    # Match exact greeting / small talk patterns
    for pattern in _GREETING_REGEXES:
        if pattern.match(cleaned):
            logger.debug("Classified '%s' as CONVERSATIONAL via pattern match", cleaned)
            return QueryIntent.CONVERSATIONAL

    # Check for general conversational intent (e.g. "Who are you?", "What can you do?")
    if _CONVERSATIONAL_INTENT_RE.search(cleaned):
        logger.debug("Classified '%s' as CONVERSATIONAL via intent match", cleaned)
        return QueryIntent.CONVERSATIONAL

    # Check if starts with a greeting and has remaining conversational text
    remainder = _LEADING_GREETING_RE.sub("", cleaned).strip()
    if not remainder:
        return QueryIntent.CONVERSATIONAL

    for pattern in _GREETING_REGEXES:
        if pattern.match(remainder):
            return QueryIntent.CONVERSATIONAL

    # Single-word checks
    words = cleaned.lower().split()
    if len(words) == 1 and words[0] in {"hi", "hello", "hey", "help", "thanks", "thx", "bye"}:
        return QueryIntent.CONVERSATIONAL

    # Default to document query for information retrieval
    return QueryIntent.DOCUMENT_QUERY


def contextualize_query(
    query: str,
    conversation_history: Sequence[dict[str, str]] | None = None,
) -> str:
    """Enhance a follow-up query with context from recent turns if pronouns/references are used.

    For example, if the previous turn asked "Tell me about Sahul Hameed"
    and the user asks "What is his GPA?", this produces
    "Tell me about Sahul Hameed | What is his GPA?" so vector retrieval
    captures the subject entity.

    Parameters
    ----------
    query:
        Current user query.
    conversation_history:
        List of message dicts with 'role' and 'content'.

    Returns
    -------
    str
        Contextualized query for vector search.
    """
    if not conversation_history:
        return query

    cleaned = query.strip()

    # Check if current query uses pronouns or continuation phrases
    has_reference = bool(_FOLLOWUP_PRONOUNS.search(cleaned))

    # Also check if it's very short (e.g., "what about education?", "and experience?")
    is_short_continuation = len(cleaned.split()) <= 4 and (
        cleaned.lower().startswith("and ")
        or cleaned.lower().startswith("what about")
        or cleaned.lower().startswith("how about")
        or has_reference
    )

    if has_reference or is_short_continuation:
        # Find the most recent user question in history
        last_user_query = None
        for msg in reversed(conversation_history):
            if msg.get("role") == "user" and msg.get("content"):
                last_user_query = msg["content"].strip()
                break

        if last_user_query and last_user_query.lower() != cleaned.lower():
            # Combine previous subject context with current question for vector retrieval
            enriched = f"{last_user_query} {cleaned}"
            logger.info(
                "Contextualized follow-up query: '%s' -> '%s'",
                cleaned,
                enriched,
            )
            return enriched

    return query
