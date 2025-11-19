"""Weekly digest generator for knowledge base insights."""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.logging import get_logger
from backend.app.models.document import Document
from backend.app.rag.claude_client import ClaudeClient, ClaudeClientError

logger = get_logger(__name__)

# Constants
MAX_SUMMARY_LENGTH = 1000  # Characters for group summaries
MAX_DOCS_PER_GROUP = 50  # Maximum documents to include in a single group summary
COMMON_KEYWORDS = {
    "meeting", "notes", "presentation", "report", "analysis", "research",
    "design", "specification", "requirements", "proposal", "review",
    "budget", "finance", "legal", "contract", "policy", "procedure",
    "technical", "documentation", "tutorial", "guide", "manual",
    "email", "correspondence", "memo", "announcement", "minutes"
}


def _get_iso_week_info(date: datetime) -> tuple[int, int, str, str]:
    """
    Get ISO week number and year for a given date.

    Args:
        date: Date to get week info for

    Returns:
        Tuple of (year, week_number, start_date_str, end_date_str)
    """
    iso_calendar = date.isocalendar()
    year = iso_calendar[0]
    week = iso_calendar[1]

    # Calculate week start (Monday) and end (Sunday)
    week_start = date - timedelta(days=date.weekday())
    week_end = week_start + timedelta(days=6)

    return year, week, week_start.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")


def _extract_keywords(text: str) -> set[str]:
    """
    Extract keywords from text for basic topic clustering.

    Args:
        text: Text to extract keywords from

    Returns:
        Set of lowercase keywords
    """
    if not text:
        return set()

    # Convert to lowercase and extract words
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())

    # Filter to common keywords
    keywords = {w for w in words if w in COMMON_KEYWORDS}

    return keywords


def _group_documents_by_topic(documents: list[Document]) -> dict[str, list[Document]]:
    """
    Group documents by topic using document type and keyword clustering.

    Args:
        documents: List of documents to group

    Returns:
        Dictionary mapping group name to list of documents
    """
    logger.info(f"Grouping {len(documents)} documents by topic")

    # Group by source type first
    type_groups = defaultdict(list)
    for doc in documents:
        type_groups[doc.source_type].append(doc)

    # Further split large groups by keywords
    final_groups = {}

    for source_type, docs in type_groups.items():
        if len(docs) <= MAX_DOCS_PER_GROUP:
            # Small group, keep as is
            group_name = source_type.replace("_", " ").title()
            final_groups[group_name] = docs
            logger.debug(f"Group '{group_name}': {len(docs)} documents")
        else:
            # Large group, split by keywords
            keyword_groups = defaultdict(list)
            uncategorized = []

            for doc in docs:
                # Extract keywords from title and path
                text = f"{doc.title or ''} {doc.path}"
                keywords = _extract_keywords(text)

                if keywords:
                    # Use first keyword as group identifier
                    keyword = sorted(keywords)[0]
                    keyword_groups[keyword].append(doc)
                else:
                    uncategorized.append(doc)

            # Add keyword-based subgroups
            for keyword, keyword_docs in keyword_groups.items():
                group_name = f"{source_type.replace('_', ' ').title()} - {keyword.title()}"
                final_groups[group_name] = keyword_docs
                logger.debug(f"Group '{group_name}': {len(keyword_docs)} documents")

            # Add uncategorized if any
            if uncategorized:
                group_name = f"{source_type.replace('_', ' ').title()} - Other"
                final_groups[group_name] = uncategorized
                logger.debug(f"Group '{group_name}': {len(uncategorized)} documents")

    logger.info(f"Created {len(final_groups)} topic groups")
    return final_groups


def _summarize_document_group(
    group_name: str,
    documents: list[Document],
    claude_client: ClaudeClient | None
) -> str:
    """
    Generate a summary for a group of documents.

    Args:
        group_name: Name of the document group
        documents: List of documents in the group
        claude_client: Claude client for AI summarization (optional)

    Returns:
        Summary text (markdown formatted)
    """
    logger.info(f"Summarizing group '{group_name}' with {len(documents)} documents")

    # Prepare document list
    doc_list = []
    for doc in documents[:MAX_DOCS_PER_GROUP]:
        title = doc.title or Path(doc.path).name
        doc_list.append(f"- {title}")

    if len(documents) > MAX_DOCS_PER_GROUP:
        doc_list.append(f"- ... and {len(documents) - MAX_DOCS_PER_GROUP} more")

    # If Claude is not available, return basic summary
    if not claude_client:
        logger.debug(f"No Claude client available, using basic summary for '{group_name}'")
        summary = f"**{group_name}** ({len(documents)} document{'s' if len(documents) != 1 else ''})\n\n"
        summary += "\n".join(doc_list)
        return summary

    # Use Claude to generate intelligent summary
    try:
        # Prepare context for Claude
        doc_info = []
        for doc in documents[:20]:  # Limit to first 20 for token efficiency
            title = doc.title or Path(doc.path).name
            path = doc.path
            modified = doc.modified_ts.strftime("%Y-%m-%d") if doc.modified_ts else "unknown"
            doc_info.append(f"- {title} ({path}) [modified: {modified}]")

        context = "\n".join(doc_info)

        prompt = f"""Please analyze this group of documents and provide:
1. A brief 2-3 sentence overview of what these documents cover
2. 3-5 key themes or topics identified across these documents
3. Any notable patterns or insights

Group: {group_name}
Number of documents: {len(documents)}

Documents:
{context}

Please format your response as:
Overview: [2-3 sentences]

Key themes:
- [theme 1]
- [theme 2]
- [theme 3]

Insights: [1-2 notable observations]
"""

        summary_text = claude_client.summarize(
            text=prompt,
            max_length=MAX_SUMMARY_LENGTH,
            temperature=0.4
        )

        # Format as markdown
        summary = f"**{group_name}** ({len(documents)} document{'s' if len(documents) != 1 else ''})\n\n"
        summary += summary_text

        logger.debug(f"Generated AI summary for '{group_name}'")
        return summary

    except ClaudeClientError as e:
        logger.warning(f"Failed to generate AI summary for '{group_name}': {e}")
        # Fallback to basic summary
        summary = f"**{group_name}** ({len(documents)} document{'s' if len(documents) != 1 else ''})\n\n"
        summary += "\n".join(doc_list)
        return summary
    except Exception as e:
        logger.error(f"Unexpected error summarizing '{group_name}': {e}", exc_info=True)
        # Fallback to basic summary
        summary = f"**{group_name}** ({len(documents)} document{'s' if len(documents) != 1 else ''})\n\n"
        summary += "\n".join(doc_list)
        return summary


def _generate_insights(
    documents: list[Document],
    groups: dict[str, list[Document]],
    claude_client: ClaudeClient | None
) -> list[str]:
    """
    Generate key insights from the document set.

    Args:
        documents: All documents in the digest
        groups: Grouped documents
        claude_client: Claude client for AI insights (optional)

    Returns:
        List of insight strings
    """
    logger.info(f"Generating insights from {len(documents)} documents")

    insights = []

    # Basic statistical insights
    insights.append(f"Added {len(documents)} new document{'s' if len(documents) != 1 else ''} this week")
    insights.append(f"Documents organized into {len(groups)} topic area{'s' if len(groups) != 1 else ''}")

    # Document type distribution
    type_counts = defaultdict(int)
    for doc in documents:
        type_counts[doc.source_type] += 1

    if type_counts:
        top_type = max(type_counts.items(), key=lambda x: x[1])
        insights.append(
            f"Most common document type: {top_type[0].replace('_', ' ').title()} "
            f"({top_type[1]} document{'s' if top_type[1] != 1 else ''})"
        )

    # Use Claude for deeper insights if available
    if claude_client and len(documents) > 0:
        try:
            # Prepare summary of groups for Claude
            group_summary = []
            for group_name, group_docs in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
                group_summary.append(f"- {group_name}: {len(group_docs)} documents")

            prompt = f"""Based on this week's knowledge base additions, identify 2-3 notable insights or patterns.
Focus on what these documents tell us about current focus areas, priorities, or trends.

Total documents: {len(documents)}
Document groups:
{chr(10).join(group_summary)}

Provide 2-3 concise bullet points (one sentence each) highlighting the most interesting insights.
Format as a simple bulleted list without any preamble.
"""

            ai_insights = claude_client.summarize(
                text=prompt,
                max_length=500,
                temperature=0.5
            )

            # Parse AI insights and add to list
            ai_bullets = [line.strip().lstrip('•-*').strip()
                         for line in ai_insights.split('\n')
                         if line.strip() and not line.strip().startswith('#')]

            insights.extend([insight for insight in ai_bullets if insight])
            logger.debug(f"Added {len(ai_bullets)} AI-generated insights")

        except Exception as e:
            logger.warning(f"Failed to generate AI insights: {e}")

    return insights


def _format_markdown_report(
    year: int,
    week: int,
    week_start: str,
    week_end: str,
    documents: list[Document],
    groups: dict[str, list[Document]],
    insights: list[str]
) -> str:
    """
    Format the digest as a markdown document.

    Args:
        year: ISO year
        week: ISO week number
        week_start: Week start date (YYYY-MM-DD)
        week_end: Week end date (YYYY-MM-DD)
        documents: All documents in digest
        groups: Grouped documents
        insights: Key insights

    Returns:
        Markdown formatted report
    """
    logger.debug(f"Formatting markdown report for {year}-W{week:02d}")

    lines = []

    # Header
    lines.append(f"# Weekly Knowledge Digest - Week {week}, {year}")
    lines.append("")
    lines.append(f"**Period:** {week_start} to {week_end}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"This week, {len(documents)} new document{'s were' if len(documents) != 1 else ' was'} "
                f"added to the knowledge base across {len(groups)} topic area{'s' if len(groups) != 1 else ''}.")
    lines.append("")

    # Key Insights
    if insights:
        lines.append("## Key Insights")
        lines.append("")
        for insight in insights:
            if not insight.startswith('-') and not insight.startswith('•'):
                lines.append(f"- {insight}")
            else:
                lines.append(insight)
        lines.append("")

    # Document Groups
    lines.append("## New Documents by Topic")
    lines.append("")

    # Sort groups by number of documents (descending)
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (group_name, group_docs) in enumerate(sorted_groups, 1):
        lines.append(f"### {i}. {group_name}")
        lines.append("")
        lines.append(f"**Count:** {len(group_docs)} document{'s' if len(group_docs) != 1 else ''}")
        lines.append("")

        # List documents
        for doc in sorted(group_docs, key=lambda d: d.modified_ts or d.created_at, reverse=True):
            title = doc.title or Path(doc.path).name
            modified = (doc.modified_ts or doc.created_at).strftime("%Y-%m-%d")
            lines.append(f"- **{title}**")
            lines.append(f"  - Path: `{doc.path}`")
            lines.append(f"  - Modified: {modified}")
            if doc.tags:
                try:
                    tags = json.loads(doc.tags)
                    if tags:
                        lines.append(f"  - Tags: {', '.join(tags)}")
                except (json.JSONDecodeError, TypeError):
                    pass

        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by SolVX Knowledge Core*")
    lines.append("")

    return "\n".join(lines)


def generate_weekly_digest(
    db: Session,
    weeks_back: int = 1
) -> dict[str, Any]:
    """
    Generate a weekly digest of new and modified documents.

    This function:
    1. Finds documents added/modified in the last N weeks
    2. Groups documents by topic using keyword clustering and document types
    3. Uses Claude AI to summarize each group (if API key available)
    4. Generates a markdown report with insights and document lists
    5. Saves the report to the insights directory

    Args:
        db: Database session
        weeks_back: Number of weeks to look back (default: 1)

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - path (str): Path to saved digest file
            - markdown (str): Generated markdown content
            - error (str, optional): Error message if failed
    """
    logger.info(f"Starting weekly digest generation (weeks_back={weeks_back})")

    try:
        # Calculate date range
        now = datetime.now()
        cutoff_date = now - timedelta(weeks=weeks_back)

        # Get ISO week info
        year, week, week_start, week_end = _get_iso_week_info(now)

        logger.info(f"Digest for week {year}-W{week:02d} ({week_start} to {week_end})")
        logger.info(f"Fetching documents modified since {cutoff_date.strftime('%Y-%m-%d')}")

        # Query documents
        try:
            stmt = (
                select(Document)
                .where(Document.status == "embedded")
                .where(
                    (Document.modified_ts >= cutoff_date) |
                    (Document.indexed_ts >= cutoff_date)
                )
                .order_by(Document.modified_ts.desc())
            )

            result = db.execute(stmt)
            documents = list(result.scalars().all())

            logger.info(f"Found {len(documents)} documents in date range")

        except Exception as e:
            logger.error(f"Database query failed: {e}", exc_info=True)
            return {
                "success": False,
                "path": "",
                "markdown": "",
                "error": f"Database query failed: {str(e)}"
            }

        # Handle empty result
        if not documents:
            logger.info("No documents found for digest period")

            # Create minimal digest
            markdown = f"""# Weekly Knowledge Digest - Week {week}, {year}

**Period:** {week_start} to {week_end}
**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

No new documents were added to the knowledge base this week.

---

*Generated by SolVX Knowledge Core*
"""

            # Save to file
            try:
                filename = f"weekly-{year}-{week:02d}.md"
                filepath = settings.insights_path / filename

                filepath.write_text(markdown, encoding="utf-8")
                logger.info(f"Saved empty digest to {filepath}")

                return {
                    "success": True,
                    "path": str(filepath),
                    "markdown": markdown
                }

            except Exception as e:
                logger.error(f"Failed to save digest file: {e}", exc_info=True)
                return {
                    "success": False,
                    "path": "",
                    "markdown": markdown,
                    "error": f"Failed to save file: {str(e)}"
                }

        # Group documents by topic
        try:
            groups = _group_documents_by_topic(documents)
        except Exception as e:
            logger.error(f"Document grouping failed: {e}", exc_info=True)
            # Fallback: create single group
            groups = {"All Documents": documents}

        # Initialize Claude client (may be None if no API key)
        claude_client = None
        if settings.claude_api_key:
            try:
                claude_client = ClaudeClient()
                logger.info("Claude client initialized for AI-powered summaries")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude client: {e}")
        else:
            logger.info("No Claude API key configured, using basic summaries")

        # Generate insights
        try:
            insights = _generate_insights(documents, groups, claude_client)
        except Exception as e:
            logger.error(f"Insight generation failed: {e}", exc_info=True)
            # Fallback to basic insights
            insights = [
                f"Added {len(documents)} new document{'s' if len(documents) != 1 else ''} this week",
                f"Documents organized into {len(groups)} topic area{'s' if len(groups) != 1 else ''}"
            ]

        # Format markdown report
        try:
            markdown = _format_markdown_report(
                year=year,
                week=week,
                week_start=week_start,
                week_end=week_end,
                documents=documents,
                groups=groups,
                insights=insights
            )
        except Exception as e:
            logger.error(f"Markdown formatting failed: {e}", exc_info=True)
            return {
                "success": False,
                "path": "",
                "markdown": "",
                "error": f"Failed to format report: {str(e)}"
            }

        # Save to file
        try:
            filename = f"weekly-{year}-{week:02d}.md"
            filepath = settings.insights_path / filename

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            filepath.write_text(markdown, encoding="utf-8")

            logger.info(f"Successfully saved digest to {filepath}")
            logger.info(f"Digest contains {len(documents)} documents in {len(groups)} groups")

            return {
                "success": True,
                "path": str(filepath),
                "markdown": markdown
            }

        except Exception as e:
            logger.error(f"Failed to save digest file: {e}", exc_info=True)
            return {
                "success": False,
                "path": "",
                "markdown": markdown,
                "error": f"Failed to save file: {str(e)}"
            }

    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Unexpected error generating weekly digest: {e}", exc_info=True)
        return {
            "success": False,
            "path": "",
            "markdown": "",
            "error": f"Unexpected error: {str(e)}"
        }
