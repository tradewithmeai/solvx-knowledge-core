"""
RAG (Retrieval Augmented Generation) prompts for knowledge base queries.

This module provides system and user prompts designed for accurate,
citation-grounded responses using retrieved context.
"""

from typing import List, Dict, Any


SYSTEM_PROMPT = """You are a knowledge assistant powered by a local knowledge base. Your role is to provide accurate, helpful answers grounded in retrieved context.

CRITICAL RULES:
1. ALWAYS cite your sources using this format: [source: filename.ext p.12]
   - Use the exact filename provided in the context
   - Include page number when available, otherwise omit p.X
   - Place citations after the relevant sentence or claim

2. ANSWER DIRECTLY - no explanations, reasoning chains, or preamble
   - Get straight to the answer
   - Structure response logically but concisely
   - One clear response, not multiple interpretations

3. SOURCE ATTRIBUTION is mandatory
   - Every factual claim must be sourced or clearly marked as inference
   - If multiple sources support a claim, cite all: [source: file1.pdf p.5, file2.md p.3]
   - Never make up information not in the context

4. ACCURACY GUARDRAILS
   - If context is insufficient, say "Information not found in knowledge base"
   - Do not speculate or hallucinate details
   - Acknowledge uncertainty: "The provided context does not specify..."
   - Only use information from provided context for factual claims

5. CONTEXT-AWARE RESPONSES
   - Use only the retrieved chunks provided
   - Synthesize information across chunks when relevant
   - Maintain consistency with provided context
   - Flag contradictions between sources if they exist

Respond with facts, not assumptions. Cite everything. Stay accurate."""


USER_PROMPT_TEMPLATE = """QUERY: {query}

RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
- Answer the query using only the retrieved context above
- Cite all sources using [source: filename.ext] format
- Provide direct, concise answers
- If information is not in the context, state that explicitly
- Do not add information beyond what is provided

ANSWER:"""


def get_system_prompt() -> str:
    """
    Retrieve the system prompt for RAG queries.

    Returns:
        str: The system prompt that defines RAG behavior and constraints
    """
    return SYSTEM_PROMPT


def format_user_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Format a user prompt with query and context chunks.

    Args:
        query (str): The user's question or query
        chunks (List[Dict[str, Any]]): Retrieved context chunks with structure:
            [
                {
                    "content": "text content",
                    "source": "filename.ext",
                    "page": 12  # optional
                },
                ...
            ]

    Returns:
        str: Formatted user prompt ready for LLM processing

    Example:
        >>> chunks = [
        ...     {"content": "Python is a programming language", "source": "guide.md"},
        ...     {"content": "It was created by Guido van Rossum", "source": "history.pdf", "page": 5}
        ... ]
        >>> prompt = format_user_prompt("What is Python?", chunks)
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        content = chunk.get("content", "")

        # Format source reference
        source_ref = f"[source: {source}"
        if page is not None:
            source_ref += f" p.{page}"
        source_ref += "]"

        # Build context entry
        context_entry = f"[{i}] {source_ref}\n{content}"
        context_parts.append(context_entry)

    # Join all context chunks with blank lines
    formatted_context = "\n\n".join(context_parts)

    # Format the complete prompt
    return USER_PROMPT_TEMPLATE.format(
        query=query,
        context=formatted_context if formatted_context else "(No relevant context found)"
    )
