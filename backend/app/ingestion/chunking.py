"""Production-ready token-aware chunking with soft sentence boundaries."""

import re
from typing import List, Optional

from backend.app.ingestion.parsers.base import ParsedBlock
from backend.app.logging import get_logger

logger = get_logger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
    logger.debug("tiktoken available for accurate token counting")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning(
        "tiktoken not available, falling back to approximate token counting. "
        "Install tiktoken for more accurate results: pip install tiktoken"
    )


class TokenCounter:
    """Token counter with tiktoken support and fallback."""

    def __init__(self, model: str = "cl100k_base"):
        """
        Initialize token counter.

        Args:
            model: tiktoken encoding model (default: cl100k_base for GPT-4/GPT-3.5)
        """
        self.encoding = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(model)
                logger.debug(f"Initialized tiktoken with {model} encoding")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}, using fallback")
                self.encoding = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        try:
            if self.encoding is not None:
                # Use tiktoken for accurate counting
                return len(self.encoding.encode(text))
            else:
                # Fallback: approximate token count
                # Rule of thumb: ~4 characters per token for English text
                # More conservative: count words and divide by 0.75
                word_count = len(text.split())
                return max(1, int(word_count / 0.75))
        except Exception as e:
            logger.error(f"Token counting failed: {e}, using word-based fallback")
            # Emergency fallback
            word_count = len(text.split())
            return max(1, int(word_count / 0.75))


class SentenceSplitter:
    """Split text into sentences with soft boundaries."""

    # Sentence boundary patterns
    # Matches: . ! ? followed by space/newline, but not abbreviations like "Dr.", "Mr.", etc.
    SENTENCE_END_PATTERN = re.compile(
        r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+(?=[A-Z])"
    )

    # Additional patterns for common abbreviations to avoid false splits
    ABBREV_PATTERN = re.compile(
        r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|etc|vs|e\.g|i\.e|Ph\.D|M\.D|U\.S|Inc|Ltd|Co)\."
    )

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """
        Split text into sentences, handling common edge cases.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []

        try:
            # Split by sentence boundaries
            sentences = cls.SENTENCE_END_PATTERN.split(text)

            # Filter out empty sentences and strip whitespace
            sentences = [s.strip() for s in sentences if s.strip()]

            # If no sentences were found (e.g., single sentence without period)
            if not sentences and text.strip():
                sentences = [text.strip()]

            return sentences

        except Exception as e:
            logger.warning(f"Sentence splitting failed: {e}, returning whole text")
            # Fallback: return the whole text as one sentence
            return [text.strip()] if text.strip() else []


def chunk_document(
    document_id: int,
    blocks: List[ParsedBlock],
    target_size: int = 800,
    overlap: int = 200,
) -> List[dict]:
    """
    Chunk document blocks into token-aware chunks with soft sentence boundaries.

    This function takes parsed document blocks and creates chunks that:
    - Target a specific token count
    - Prefer to break at sentence boundaries when possible
    - Maintain overlap between chunks for context continuity
    - Preserve complete provenance (document_id, sequence, page, section, offsets)

    Args:
        document_id: ID of the document being chunked
        blocks: List of ParsedBlock objects from parser
        target_size: Target chunk size in tokens (default: 800)
        overlap: Overlap between chunks in tokens (default: 200)

    Returns:
        List of chunk dictionaries with fields:
            - document_id: Document ID
            - seq: Sequence number (0-indexed)
            - text: Chunk text content
            - token_count: Actual token count
            - page: Page number (if available)
            - section: Section name (if available)
            - offset_start: Byte offset start in original document (if available)
            - offset_end: Byte offset end in original document (if available)

    Edge cases handled:
        - Empty blocks list: returns empty list
        - Very short texts: returns single chunk
        - Very long sentences: splits mid-sentence if necessary
        - Missing metadata: uses None for optional fields
        - Token counting failures: uses fallback method
    """
    try:
        logger.info(
            f"Starting chunking for document_id={document_id}, "
            f"blocks={len(blocks)}, target_size={target_size}, overlap={overlap}"
        )

        # Validate inputs
        if not blocks:
            logger.warning(f"No blocks provided for document_id={document_id}")
            return []

        if target_size <= 0:
            logger.error(f"Invalid target_size={target_size}, using default 800")
            target_size = 800

        if overlap < 0:
            logger.error(f"Invalid overlap={overlap}, using default 200")
            overlap = 200

        if overlap >= target_size:
            logger.warning(
                f"Overlap ({overlap}) >= target_size ({target_size}), "
                f"reducing overlap to {target_size // 2}"
            )
            overlap = target_size // 2

        # Initialize token counter and sentence splitter
        token_counter = TokenCounter()
        sentence_splitter = SentenceSplitter()

        chunks: List[dict] = []
        current_chunk_sentences: List[str] = []
        current_chunk_tokens: int = 0
        current_block_idx: int = 0
        chunk_seq: int = 0

        # Track provenance for current chunk
        chunk_page: Optional[int] = None
        chunk_section: Optional[str] = None
        chunk_offset_start: Optional[int] = None
        chunk_offset_end: Optional[int] = None

        def create_chunk() -> Optional[dict]:
            """Create a chunk from current sentences and reset state."""
            nonlocal current_chunk_sentences, current_chunk_tokens
            nonlocal chunk_page, chunk_section, chunk_offset_start, chunk_offset_end
            nonlocal chunk_seq

            if not current_chunk_sentences:
                return None

            chunk_text = " ".join(current_chunk_sentences)
            if not chunk_text.strip():
                return None

            # Calculate actual token count for the final chunk
            actual_tokens = token_counter.count_tokens(chunk_text)

            chunk = {
                "document_id": document_id,
                "seq": chunk_seq,
                "text": chunk_text,
                "token_count": actual_tokens,
                "page": chunk_page,
                "section": chunk_section,
                "offset_start": chunk_offset_start,
                "offset_end": chunk_offset_end,
            }

            logger.debug(
                f"Created chunk seq={chunk_seq}, tokens={actual_tokens}, "
                f"page={chunk_page}, section={chunk_section}"
            )

            chunk_seq += 1

            # Keep overlap sentences for next chunk
            if overlap > 0 and len(current_chunk_sentences) > 1:
                # Calculate how many sentences to keep for overlap
                overlap_sentences = []
                overlap_tokens = 0

                # Work backwards from the end
                for sentence in reversed(current_chunk_sentences):
                    sentence_tokens = token_counter.count_tokens(sentence)
                    if overlap_tokens + sentence_tokens <= overlap:
                        overlap_sentences.insert(0, sentence)
                        overlap_tokens += sentence_tokens
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = overlap_tokens
            else:
                current_chunk_sentences = []
                current_chunk_tokens = 0

            return chunk

        # Process each block
        for block_idx, block in enumerate(blocks):
            try:
                if not block.text or not block.text.strip():
                    logger.debug(f"Skipping empty block at index {block_idx}")
                    continue

                # Split block into sentences
                sentences = sentence_splitter.split_sentences(block.text)

                if not sentences:
                    logger.debug(f"No sentences extracted from block {block_idx}")
                    continue

                logger.debug(
                    f"Processing block {block_idx}: {len(sentences)} sentences, "
                    f"page={block.page}, section={block.section}"
                )

                # Process each sentence
                for sentence in sentences:
                    sentence_tokens = token_counter.count_tokens(sentence)

                    # Handle very long sentences that exceed target size
                    if sentence_tokens > target_size:
                        logger.warning(
                            f"Sentence exceeds target_size ({sentence_tokens} > {target_size}), "
                            f"will split mid-sentence"
                        )

                        # Finalize current chunk if it has content
                        if current_chunk_sentences:
                            chunk = create_chunk()
                            if chunk:
                                chunks.append(chunk)

                        # Split long sentence by words
                        words = sentence.split()
                        current_words = []
                        current_word_tokens = 0

                        for word in words:
                            word_tokens = token_counter.count_tokens(word + " ")
                            if current_word_tokens + word_tokens > target_size and current_words:
                                # Create chunk from current words
                                chunk_text = " ".join(current_words)
                                actual_tokens = token_counter.count_tokens(chunk_text)

                                chunk = {
                                    "document_id": document_id,
                                    "seq": chunk_seq,
                                    "text": chunk_text,
                                    "token_count": actual_tokens,
                                    "page": block.page,
                                    "section": block.section,
                                    "offset_start": block.offset_start,
                                    "offset_end": block.offset_end,
                                }
                                chunks.append(chunk)
                                chunk_seq += 1

                                # Keep overlap
                                if overlap > 0:
                                    overlap_words = []
                                    overlap_tokens = 0
                                    for w in reversed(current_words):
                                        w_tokens = token_counter.count_tokens(w + " ")
                                        if overlap_tokens + w_tokens <= overlap:
                                            overlap_words.insert(0, w)
                                            overlap_tokens += w_tokens
                                        else:
                                            break
                                    current_words = overlap_words
                                    current_word_tokens = overlap_tokens
                                else:
                                    current_words = []
                                    current_word_tokens = 0
                            else:
                                current_words.append(word)
                                current_word_tokens += word_tokens

                        # Handle remaining words
                        if current_words:
                            current_chunk_sentences = [" ".join(current_words)]
                            current_chunk_tokens = token_counter.count_tokens(
                                current_chunk_sentences[0]
                            )
                            chunk_page = block.page
                            chunk_section = block.section
                            chunk_offset_start = block.offset_start
                            chunk_offset_end = block.offset_end

                    else:
                        # Normal sentence processing
                        # Check if adding this sentence would exceed target
                        if current_chunk_tokens + sentence_tokens > target_size and current_chunk_sentences:
                            # Finalize current chunk
                            chunk = create_chunk()
                            if chunk:
                                chunks.append(chunk)

                        # Add sentence to current chunk
                        if not current_chunk_sentences:
                            # Starting new chunk, set provenance
                            chunk_page = block.page
                            chunk_section = block.section
                            chunk_offset_start = block.offset_start
                            chunk_offset_end = block.offset_end

                        current_chunk_sentences.append(sentence)
                        current_chunk_tokens += sentence_tokens

            except Exception as e:
                logger.error(
                    f"Error processing block {block_idx} for document_id={document_id}: {e}",
                    exc_info=True,
                )
                # Continue processing remaining blocks instead of crashing
                continue

        # Finalize any remaining chunk
        if current_chunk_sentences:
            chunk = create_chunk()
            if chunk:
                chunks.append(chunk)

        logger.info(
            f"Chunking complete for document_id={document_id}: "
            f"created {len(chunks)} chunks from {len(blocks)} blocks"
        )

        # Log statistics
        if chunks:
            total_tokens = sum(c["token_count"] for c in chunks)
            avg_tokens = total_tokens / len(chunks)
            min_tokens = min(c["token_count"] for c in chunks)
            max_tokens = max(c["token_count"] for c in chunks)

            logger.info(
                f"Chunk statistics: avg_tokens={avg_tokens:.1f}, "
                f"min_tokens={min_tokens}, max_tokens={max_tokens}, "
                f"total_tokens={total_tokens}"
            )

        return chunks

    except Exception as e:
        logger.error(
            f"Critical error in chunk_document for document_id={document_id}: {e}",
            exc_info=True,
        )
        # Never crash - return empty list on catastrophic failure
        return []
