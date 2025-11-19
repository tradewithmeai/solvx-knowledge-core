"""Claude API client for RAG chat and summarization with streaming support."""

import logging
import time
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator

import anthropic
from anthropic import APIError, APITimeoutError, RateLimitError

from ..config import settings

# Import prompts (will be created in prompts.py)
try:
    from .prompts import RAG_SYSTEM_PROMPT, SUMMARIZATION_PROMPT
except ImportError:
    # Fallback prompts if prompts.py doesn't exist yet
    RAG_SYSTEM_PROMPT = """You are a helpful AI assistant with access to the user's knowledge base.
Use the provided context to answer questions accurately and concisely.
If the context doesn't contain enough information, say so clearly."""

    SUMMARIZATION_PROMPT = """Summarize the following text concisely in {max_length} characters or less.
Focus on the key points and main ideas.

Text to summarize:
{text}

Summary:"""

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 30  # seconds
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7

# Token tracking storage (in production, use Redis or database)
_token_usage_cache = {
    "date": None,
    "chat_tokens": 0,
    "embed_tokens": 0,
}


class ClaudeClientError(Exception):
    """Base exception for Claude client errors."""
    pass


class TokenLimitExceededError(ClaudeClientError):
    """Raised when daily token limit is exceeded."""
    pass


class ClaudeClient:
    """Client for interacting with Claude API with streaming support and token management."""

    def __init__(
        self,
        api_key: str | None = None,
        max_tokens_per_day: int | None = None,
    ):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key (defaults to settings.claude_api_key)
            max_tokens_per_day: Maximum tokens per day (defaults to settings.max_chat_tokens_per_day)
        """
        self.api_key = api_key or settings.claude_api_key
        if not self.api_key:
            raise ClaudeClientError(
                "Claude API key not provided. Set CLAUDE_API_KEY in environment or .env file."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.max_tokens_per_day = max_tokens_per_day or settings.max_chat_tokens_per_day

        logger.info(
            f"Claude client initialized with model {DEFAULT_MODEL}, "
            f"daily limit: {self.max_tokens_per_day:,} tokens"
        )

    def _reset_daily_limits_if_needed(self) -> None:
        """Reset token counters if it's a new day."""
        today = datetime.now().date()
        if _token_usage_cache["date"] != today:
            logger.info(
                f"New day detected. Resetting token counters. "
                f"Previous usage: {_token_usage_cache['chat_tokens']:,} chat tokens"
            )
            _token_usage_cache["date"] = today
            _token_usage_cache["chat_tokens"] = 0
            _token_usage_cache["embed_tokens"] = 0

    def _check_token_limit(self, estimated_tokens: int = 0) -> None:
        """
        Check if we're within daily token limits.

        Args:
            estimated_tokens: Estimated tokens for the upcoming request

        Raises:
            TokenLimitExceededError: If daily limit would be exceeded
        """
        self._reset_daily_limits_if_needed()

        current_usage = _token_usage_cache["chat_tokens"]
        projected_usage = current_usage + estimated_tokens

        if projected_usage > self.max_tokens_per_day:
            raise TokenLimitExceededError(
                f"Daily token limit would be exceeded. "
                f"Current: {current_usage:,}, "
                f"Estimated: {estimated_tokens:,}, "
                f"Limit: {self.max_tokens_per_day:,}"
            )

        logger.debug(
            f"Token check passed: {current_usage:,}/{self.max_tokens_per_day:,} "
            f"(+{estimated_tokens:,} estimated)"
        )

    def _update_token_usage(self, tokens_used: int) -> None:
        """
        Update token usage tracking.

        Args:
            tokens_used: Number of tokens used in the request
        """
        self._reset_daily_limits_if_needed()
        _token_usage_cache["chat_tokens"] += tokens_used

        logger.info(
            f"Token usage updated: +{tokens_used:,} tokens, "
            f"total today: {_token_usage_cache['chat_tokens']:,}/{self.max_tokens_per_day:,}"
        )

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def _format_context(self, context_chunks: list[dict]) -> str:
        """
        Format context chunks for the RAG prompt.

        Args:
            context_chunks: List of context chunks with 'content' and metadata

        Returns:
            Formatted context string
        """
        if not context_chunks:
            return "No relevant context available."

        formatted_chunks = []
        for i, chunk in enumerate(context_chunks, 1):
            content = chunk.get("content", "")
            source = chunk.get("source", "Unknown")
            score = chunk.get("score", 0.0)

            formatted_chunks.append(
                f"[Context {i}] (Source: {source}, Relevance: {score:.2f})\n{content}"
            )

        return "\n\n---\n\n".join(formatted_chunks)

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            ClaudeClientError: If all retries fail
        """
        delay = INITIAL_RETRY_DELAY

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                if attempt == MAX_RETRIES:
                    logger.error(f"Rate limit exceeded after {MAX_RETRIES} attempts")
                    raise ClaudeClientError(f"Rate limit exceeded: {e}") from e

                logger.warning(
                    f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}), "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
                delay = min(delay * 2, MAX_RETRY_DELAY)

            except APITimeoutError as e:
                if attempt == MAX_RETRIES:
                    logger.error(f"API timeout after {MAX_RETRIES} attempts")
                    raise ClaudeClientError(f"API timeout: {e}") from e

                logger.warning(
                    f"API timeout (attempt {attempt}/{MAX_RETRIES}), "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
                delay = min(delay * 2, MAX_RETRY_DELAY)

            except APIError as e:
                if attempt == MAX_RETRIES:
                    logger.error(f"API error after {MAX_RETRIES} attempts: {e}")
                    raise ClaudeClientError(f"API error: {e}") from e

                # Check if error is retryable
                if hasattr(e, "status_code") and e.status_code >= 500:
                    logger.warning(
                        f"Server error {e.status_code} (attempt {attempt}/{MAX_RETRIES}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, MAX_RETRY_DELAY)
                else:
                    # Client error, don't retry
                    raise ClaudeClientError(f"API error: {e}") from e

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise ClaudeClientError(f"Unexpected error: {e}") from e

    def chat_stream(
        self,
        query: str,
        context_chunks: list[dict],
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream chat responses with RAG context.

        Args:
            query: User's query
            context_chunks: List of relevant context chunks from knowledge base
            model: Claude model to use
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Custom system prompt (defaults to RAG_SYSTEM_PROMPT)

        Yields:
            Text chunks from Claude's streaming response

        Raises:
            TokenLimitExceededError: If daily token limit would be exceeded
            ClaudeClientError: For API errors or failures
        """
        logger.info(
            f"Starting chat stream: query='{query[:50]}...', "
            f"context_chunks={len(context_chunks)}, model={model}"
        )

        # Format context
        formatted_context = self._format_context(context_chunks)

        # Estimate tokens for this request
        estimated_tokens = self._estimate_tokens(query + formatted_context) + max_tokens

        # Check token limits
        self._check_token_limit(estimated_tokens)

        # Prepare messages
        user_message = f"""Context from knowledge base:

{formatted_context}

---

User question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information, acknowledge this limitation."""

        messages = [{"role": "user", "content": user_message}]

        # Use custom or default system prompt
        system = system_prompt or RAG_SYSTEM_PROMPT

        try:
            logger.debug(f"Calling Claude API with streaming for model {model}")

            def _stream():
                """Internal function to handle streaming with retry logic."""
                return self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=messages,
                    stream=True,
                )

            # Execute with retry logic
            stream = self._retry_with_backoff(_stream)

            # Track tokens and stream response
            total_tokens = 0
            text_content = ""

            for event in stream:
                if event.type == "content_block_start":
                    logger.debug("Stream started")

                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        chunk = event.delta.text
                        text_content += chunk
                        yield chunk

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        total_tokens = event.usage.output_tokens
                        logger.debug(f"Message delta usage: {total_tokens} tokens")

                elif event.type == "message_stop":
                    logger.debug("Stream completed")
                    # Final token update
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        total_tokens = (
                            event.message.usage.input_tokens
                            + event.message.usage.output_tokens
                        )

            # Update token usage
            if total_tokens > 0:
                self._update_token_usage(total_tokens)
            else:
                # Fallback estimation if usage not provided
                estimated = self._estimate_tokens(text_content + user_message)
                self._update_token_usage(estimated)
                logger.warning(
                    f"Token usage not provided by API, using estimate: {estimated}"
                )

            logger.info(
                f"Chat stream completed successfully. Tokens used: {total_tokens}"
            )

        except TokenLimitExceededError:
            raise
        except Exception as e:
            logger.error(f"Error during chat streaming: {e}")
            raise ClaudeClientError(f"Failed to stream chat response: {e}") from e

    def summarize(
        self,
        text: str,
        max_length: int = 500,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Summarize text using Claude.

        Args:
            text: Text to summarize
            max_length: Maximum length of summary in characters
            model: Claude model to use
            temperature: Sampling temperature (lower for more focused summaries)
            max_tokens: Maximum tokens to generate

        Returns:
            Summary text

        Raises:
            TokenLimitExceededError: If daily token limit would be exceeded
            ClaudeClientError: For API errors or failures
        """
        logger.info(
            f"Starting summarization: text_length={len(text)}, "
            f"max_length={max_length}, model={model}"
        )

        # Check if text is too short to summarize
        if len(text) < max_length:
            logger.info("Text shorter than max_length, returning as-is")
            return text

        # Estimate tokens
        estimated_tokens = self._estimate_tokens(text) + max_tokens

        # Check token limits
        self._check_token_limit(estimated_tokens)

        # Prepare prompt
        prompt = SUMMARIZATION_PROMPT.format(max_length=max_length, text=text)

        messages = [{"role": "user", "content": prompt}]

        try:
            logger.debug(f"Calling Claude API for summarization with model {model}")

            def _create_message():
                """Internal function to create message with retry logic."""
                return self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                )

            # Execute with retry logic
            response = self._retry_with_backoff(_create_message)

            # Extract summary
            summary = response.content[0].text

            # Update token usage
            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            self._update_token_usage(total_tokens)

            logger.info(
                f"Summarization completed successfully. "
                f"Original: {len(text)} chars, Summary: {len(summary)} chars, "
                f"Tokens used: {total_tokens}"
            )

            return summary

        except TokenLimitExceededError:
            raise
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise ClaudeClientError(f"Failed to summarize text: {e}") from e

    def get_token_usage(self) -> dict:
        """
        Get current token usage statistics.

        Returns:
            Dictionary with token usage information
        """
        self._reset_daily_limits_if_needed()

        return {
            "date": str(_token_usage_cache["date"]),
            "chat_tokens_used": _token_usage_cache["chat_tokens"],
            "chat_tokens_limit": self.max_tokens_per_day,
            "chat_tokens_remaining": max(
                0, self.max_tokens_per_day - _token_usage_cache["chat_tokens"]
            ),
            "usage_percentage": (
                _token_usage_cache["chat_tokens"] / self.max_tokens_per_day * 100
                if self.max_tokens_per_day > 0
                else 0
            ),
        }


# Convenience function for backward compatibility
def chat_stream(
    query: str,
    context_chunks: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Generator[str, None, None]:
    """
    Convenience function for streaming chat with RAG context.

    Args:
        query: User's query
        context_chunks: List of relevant context chunks from knowledge base
        model: Claude model to use
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate

    Yields:
        Text chunks from Claude's streaming response
    """
    client = ClaudeClient()
    yield from client.chat_stream(
        query=query,
        context_chunks=context_chunks,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def summarize(
    text: str,
    max_length: int = 500,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Convenience function for text summarization.

    Args:
        text: Text to summarize
        max_length: Maximum length of summary in characters
        model: Claude model to use

    Returns:
        Summary text
    """
    client = ClaudeClient()
    return client.summarize(text=text, max_length=max_length, model=model)
