"""Reflection prompt generator for knowledge base insights."""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models.document import Document
from backend.app.models.insight import Insight
from backend.app.rag.claude_client import ClaudeClient, ClaudeClientError

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DAYS_BACK = 14
MIN_REFLECTION_PROMPTS = 3
MAX_REFLECTION_PROMPTS = 5
REFLECTION_SYSTEM_PROMPT = """You are a thoughtful knowledge base analyst.
Your role is to generate insightful, actionable reflection prompts based on a user's knowledge collection.

Generate reflection prompts that are:
1. Thought-provoking about their knowledge patterns and interests
2. Actionable with specific suggestions (e.g., "Consider exploring X based on your recent interest in Y")
3. Personal and grounded in their actual knowledge base patterns
4. Encouraging deeper thinking and exploration

Keep each prompt concise (1-2 sentences). Focus on patterns, trends, and opportunities for learning."""


def _analyze_document_trends(db: Session, days_back: int) -> dict[str, Any]:
    """
    Analyze document trends over the last N days.

    Args:
        db: Database session
        days_back: Number of days to analyze

    Returns:
        Dictionary with trend analysis
    """
    logger.debug(f"Analyzing document trends for last {days_back} days")

    try:
        # Calculate cutoff date
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        logger.debug(f"Cutoff date: {cutoff_date}")

        # Get documents created/indexed in the period
        recent_docs = db.query(Document).filter(
            Document.created_at >= cutoff_date
        ).all()

        logger.info(f"Found {len(recent_docs)} documents in period")

        # Analyze document count
        doc_count = len(recent_docs)

        # Analyze source types
        source_types: dict[str, int] = {}
        for doc in recent_docs:
            source_type = doc.source_type or "unknown"
            source_types[source_type] = source_types.get(source_type, 0) + 1

        # Analyze tags/topics
        topics: dict[str, int] = {}
        doc_titles: list[str] = []

        for doc in recent_docs:
            # Parse tags if stored as JSON
            if doc.tags:
                try:
                    tags = json.loads(doc.tags) if isinstance(doc.tags, str) else doc.tags
                    if isinstance(tags, list):
                        for tag in tags:
                            topics[tag] = topics.get(tag, 0) + 1
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"Could not parse tags for doc {doc.id}: {e}")

            # Collect titles for analysis
            if doc.title:
                doc_titles.append(doc.title)

        # Sort topics by frequency
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        top_topics = sorted_topics[:10]  # Top 10 topics

        # Get total documents in knowledge base
        total_docs = db.query(Document).count()

        trends = {
            "period_days": days_back,
            "new_documents": doc_count,
            "total_documents": total_docs,
            "source_types": source_types,
            "top_topics": top_topics,
            "top_topic_names": [topic for topic, _ in top_topics],
            "unique_topics_count": len(topics),
            "sample_titles": doc_titles[:5],
        }

        logger.info(f"Trends analysis complete: {doc_count} new docs, "
                    f"{len(source_types)} source types, {len(topics)} unique topics")

        return trends

    except Exception as e:
        logger.error(f"Error analyzing document trends: {e}", exc_info=True)
        return {
            "period_days": days_back,
            "new_documents": 0,
            "total_documents": 0,
            "source_types": {},
            "top_topics": [],
            "top_topic_names": [],
            "unique_topics_count": 0,
            "sample_titles": [],
            "error": str(e),
        }


def _build_reflection_context(trends: dict[str, Any]) -> str:
    """
    Build context string for Claude based on analyzed trends.

    Args:
        trends: Analyzed document trends

    Returns:
        Formatted context string
    """
    logger.debug("Building reflection context from trends")

    context_parts = []

    # Basic stats
    new_docs = trends.get("new_documents", 0)
    total_docs = trends.get("total_documents", 0)
    context_parts.append(f"Knowledge base has {total_docs} total documents.")
    context_parts.append(f"In the last {trends.get('period_days', 14)} days, {new_docs} new documents were added.")

    # Source types
    source_types = trends.get("source_types", {})
    if source_types:
        types_str = ", ".join([f"{count} {stype}" for stype, count in source_types.items()])
        context_parts.append(f"Document types: {types_str}.")

    # Topics
    top_topics = trends.get("top_topic_names", [])
    if top_topics:
        topics_str = ", ".join(top_topics[:5])
        context_parts.append(f"Key topics include: {topics_str}.")

    # Sample content
    sample_titles = trends.get("sample_titles", [])
    if sample_titles:
        titles_str = ", ".join([f'"{title}"' for title in sample_titles[:3]])
        context_parts.append(f"Recent document titles: {titles_str}.")

    context = "\n".join(context_parts)
    logger.debug(f"Built context: {len(context)} chars")

    return context


def _generate_reflection_prompts(
    claude_client: ClaudeClient,
    trends: dict[str, Any]
) -> list[str]:
    """
    Use Claude to generate reflection prompts based on trends.

    Args:
        claude_client: Initialized Claude client
        trends: Analyzed document trends

    Returns:
        List of reflection prompt strings, or empty list on error
    """
    logger.info("Generating reflection prompts with Claude")

    try:
        # Build context
        context = _build_reflection_context(trends)

        # Build prompt for Claude
        user_message = f"""Based on this user's knowledge base activity, generate {MIN_REFLECTION_PROMPTS}-{MAX_REFLECTION_PROMPTS}
reflection prompts or questions that will help them think deeper about their knowledge and learning patterns.

Knowledge Base Context:
{context}

Please provide exactly {MIN_REFLECTION_PROMPTS}-{MAX_REFLECTION_PROMPTS} reflection prompts, one per line.
Each prompt should be thought-provoking and actionable."""

        logger.debug(f"User message length: {len(user_message)} chars")

        # Call Claude API
        messages = [{"role": "user", "content": user_message}]

        logger.debug("Calling Claude API to generate reflections")
        response = claude_client.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.7,
            system=REFLECTION_SYSTEM_PROMPT,
            messages=messages,
        )

        # Extract response
        response_text = response.content[0].text
        logger.debug(f"Claude response: {len(response_text)} chars")

        # Parse response into prompts (one per line)
        prompts = [
            line.strip()
            for line in response_text.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Filter out empty lines and ensure we have valid prompts
        prompts = [p for p in prompts if len(p) > 10]  # Filter very short lines

        # Ensure we have between MIN and MAX prompts
        if len(prompts) < MIN_REFLECTION_PROMPTS:
            logger.warning(f"Generated only {len(prompts)} prompts, expected at least {MIN_REFLECTION_PROMPTS}")
        if len(prompts) > MAX_REFLECTION_PROMPTS:
            logger.info(f"Generated {len(prompts)} prompts, truncating to {MAX_REFLECTION_PROMPTS}")
            prompts = prompts[:MAX_REFLECTION_PROMPTS]

        logger.info(f"Successfully generated {len(prompts)} reflection prompts")

        # Log token usage
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        logger.info(f"Claude API used {tokens_used} tokens")

        return prompts

    except ClaudeClientError as e:
        logger.error(f"Claude API error generating reflections: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error generating reflection prompts: {e}", exc_info=True)
        return []


def _store_reflection_insight(
    db: Session,
    prompts: list[str],
    trends: dict[str, Any]
) -> bool:
    """
    Store reflection insight in the database.

    Args:
        db: Database session
        prompts: List of reflection prompts
        trends: Analyzed trends

    Returns:
        True if stored successfully, False otherwise
    """
    logger.info(f"Storing reflection insight with {len(prompts)} prompts")

    try:
        # Build markdown content
        markdown_content = "# Reflection Prompts\n\n"
        markdown_content += "Consider these questions about your knowledge and learning patterns:\n\n"

        for i, prompt in enumerate(prompts, 1):
            markdown_content += f"{i}. {prompt}\n\n"

        # Add metadata
        markdown_content += "---\n\n"
        markdown_content += "**Analysis Period**: Last 14 days\n"
        markdown_content += f"**New Documents**: {trends.get('new_documents', 0)}\n"
        markdown_content += f"**Total Documents**: {trends.get('total_documents', 0)}\n"

        top_topics = trends.get("top_topic_names", [])
        if top_topics:
            markdown_content += f"**Key Topics**: {', '.join(top_topics[:5])}\n"

        # Create insight record
        insight = Insight(
            type="reflection",
            period_start=datetime.utcnow() - timedelta(days=trends.get("period_days", 14)),
            period_end=datetime.utcnow(),
            markdown=markdown_content,
        )

        db.add(insight)
        db.commit()

        logger.info(f"Reflection insight stored with id={insight.id}")
        return True

    except Exception as e:
        logger.error(f"Error storing reflection insight: {e}", exc_info=True)
        db.rollback()
        return False


def generate_reflection(
    db: Session,
    days_back: int = DEFAULT_DAYS_BACK,
) -> dict[str, Any]:
    """
    Generate reflection prompts based on knowledge base analysis.

    Analyzes document trends over the specified period and uses Claude to generate
    thoughtful, actionable reflection prompts. Results are stored in the insights table.

    Args:
        db: SQLAlchemy database session
        days_back: Number of days to analyze (default: 14)

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - prompts (list[str]): Generated reflection prompts (empty if failed)
            - message (str): Status message
            - error (str|None): Error message if any
    """
    logger.info(f"Starting reflection generation (days_back={days_back})")

    try:
        # Validate days_back
        if days_back < 1:
            days_back = DEFAULT_DAYS_BACK
            logger.warning(f"Invalid days_back, using default: {days_back}")

        # Check if reflection is enabled
        if not settings.reflection_enabled:
            logger.info("Reflection generation is disabled in settings")
            return {
                "success": False,
                "prompts": [],
                "message": "Reflection generation is disabled",
                "error": None,
            }

        # Check if Claude API key is configured
        if not settings.claude_api_key:
            logger.error("Claude API key not configured")
            return {
                "success": False,
                "prompts": [],
                "message": "Claude API key not configured",
                "error": "CLAUDE_API_KEY environment variable not set",
            }

        # Step 1: Analyze trends
        logger.info("Step 1: Analyzing document trends")
        trends = _analyze_document_trends(db, days_back)

        # Check if we have any data to work with
        if trends.get("new_documents", 0) == 0:
            logger.info("No new documents in the analysis period")
            # Still generate reflections for existing knowledge base
            logger.info("Generating reflections based on overall knowledge base")

        # Step 2: Generate prompts with Claude
        logger.info("Step 2: Generating reflection prompts")
        try:
            claude_client = ClaudeClient()
        except ClaudeClientError as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            return {
                "success": False,
                "prompts": [],
                "message": "Failed to initialize Claude client",
                "error": str(e),
            }

        prompts = _generate_reflection_prompts(claude_client, trends)

        if not prompts:
            logger.warning("No reflection prompts generated")
            return {
                "success": False,
                "prompts": [],
                "message": "Failed to generate reflection prompts",
                "error": "Claude API returned empty response",
            }

        # Step 3: Store in database
        logger.info("Step 3: Storing reflection in database")
        stored_successfully = _store_reflection_insight(db, prompts, trends)

        if not stored_successfully:
            logger.error("Failed to store reflection insight")
            return {
                "success": False,
                "prompts": prompts,  # Return prompts even if storage failed
                "message": "Generated prompts but failed to store in database",
                "error": "Database storage failed",
            }

        # Success
        logger.info(f"Reflection generation completed successfully with {len(prompts)} prompts")
        return {
            "success": True,
            "prompts": prompts,
            "message": f"Generated {len(prompts)} reflection prompts",
            "error": None,
        }

    except Exception as e:
        logger.error(f"Unexpected error in reflection generation: {e}", exc_info=True)
        return {
            "success": False,
            "prompts": [],
            "message": "Unexpected error during reflection generation",
            "error": str(e),
        }
