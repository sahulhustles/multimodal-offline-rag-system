"""Prompt builder for grounded answer generation and interactive conversation.

Constructs system prompts and messages for Ollama /api/chat.
Supports two modes:
1. Grounded mode (build_messages): strict source citation and anti-hallucination rules.
2. Conversational mode (build_conversational_messages): friendly, interactive dialogue,
   explaining capabilities and guiding user queries across multimodal documents.
"""

from __future__ import annotations

from typing import Sequence

from backend.generation.context_assembler import AssembledContext
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# The exact phrase the LLM must return when evidence is insufficient for a document query.
INSUFFICIENT_CONTEXT_PHRASE = (
    "I don't have enough information in the selected sources to answer that."
)

GROUNDED_SYSTEM_PROMPT_TEMPLATE = """\
You are an intelligent, grounded assistant for an offline multimodal document system.

STRICT RULES — you must follow every rule without exception:

1. Answer ONLY from the provided source documents below. Do NOT use outside knowledge.
2. Cite every factual statement using the source IDs provided (e.g. [S1], [S2], [S1][S3]).
3. You may ONLY cite source IDs that appear in the provided context: {allowed_ids}.
4. Do NOT invent, fabricate, or hallucinate any citation, source ID, page number, timestamp, or quote.
5. If the provided sources do not contain enough information to answer the question, respond with EXACTLY this phrase and nothing else:
   "{insufficient_phrase}"
6. Provide an informative, well-structured, helpful answer with clear formatting.
7. Be direct and helpful while maintaining strict citation accuracy.

SOURCE DOCUMENTS:

{context}
"""

CONVERSATIONAL_SYSTEM_PROMPT = """\
You are the intelligent assistant for the Offline Multimodal RAG System.
You interact warmly, naturally, and helpfully with users.

About your capabilities:
- You help users search, analyze, and query offline multimodal documents (PDFs, Word documents, images, screenshots, audio recordings).
- You can explain document contents, summarize records, and answer questions grounded strictly in user-uploaded files.
- You can answer greetings, help requests, questions about your capabilities, and clarify previous answers.

Guidelines:
- When greeted, respond warmly and briefly invite the user to ask questions about their uploaded documents.
- When asked who you are or what you can do, concisely explain your multimodal capabilities (PDFs, images, audio transcripts, certificates).
- Keep conversational responses concise, clear, and engaging.
- Do NOT generate [S#] citations for general conversational greetings or system questions.
"""


def build_messages(
    query: str,
    assembled_context: AssembledContext,
    conversation_history: Sequence[dict[str, str]] | None = None,
    max_history_turns: int = 4,
) -> list[dict[str, str]]:
    """Build the chat messages (system + history + user) for grounded document answering.

    Parameters
    ----------
    query:
        The user's natural-language question.
    assembled_context:
        The assembled context from ``assemble_context()``.
    conversation_history:
        Optional sequence of previous messages with 'role' and 'content'.
    max_history_turns:
        Maximum recent turns to include for context.

    Returns
    -------
    list[dict[str, str]]
        Messages in Ollama chat format: [{role, content}, ...].
    """
    allowed_ids = ", ".join(sorted(assembled_context.source_id_set))
    context_text = assembled_context.render()

    system_content = GROUNDED_SYSTEM_PROMPT_TEMPLATE.format(
        allowed_ids=allowed_ids,
        insufficient_phrase=INSUFFICIENT_CONTEXT_PHRASE,
        context=context_text,
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

    # Append recent conversation history if provided
    if conversation_history:
        recent = conversation_history[-max_history_turns:]
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

    # Current user query
    messages.append({"role": "user", "content": query})

    logger.info(
        "Grounded prompt built: system=%d chars, history=%d msgs, user=%d chars, allowed_ids=%s",
        len(system_content),
        len(messages) - 2,
        len(query),
        allowed_ids,
    )

    return messages


def build_conversational_messages(
    query: str,
    conversation_history: Sequence[dict[str, str]] | None = None,
    max_history_turns: int = 6,
) -> list[dict[str, str]]:
    """Build chat messages for interactive conversational dialogue (greetings, help, chit-chat).

    Parameters
    ----------
    query:
        The user's conversational message.
    conversation_history:
        Optional sequence of previous messages with 'role' and 'content'.
    max_history_turns:
        Maximum recent turns to include.

    Returns
    -------
    list[dict[str, str]]
        Messages for Ollama /api/chat.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": CONVERSATIONAL_SYSTEM_PROMPT}
    ]

    if conversation_history:
        recent = conversation_history[-max_history_turns:]
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": query})

    logger.info(
        "Conversational prompt built: history=%d msgs, user=%d chars",
        len(messages) - 2,
        len(query),
    )

    return messages
