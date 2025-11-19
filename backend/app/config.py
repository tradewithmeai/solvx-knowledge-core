"""Configuration management with environment variables and validation."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "SolVX Knowledge Core"
    app_version: str = "0.1.0"
    debug: bool = False

    # Server
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8787, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")

    # Data directories
    solvx_data_dir: Path = Field(default=Path("./data"), description="Root data directory")
    solvx_db_path: Path = Field(
        default=Path("./data/solvx.db"), description="SQLite database path"
    )

    # Embedding provider
    embed_provider: Literal["openai", "cohere", "voyage", "local"] = Field(
        default="local", description="Embedding provider to use"
    )

    # API Keys (optional, depending on provider)
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    cohere_api_key: str | None = Field(default=None, description="Cohere API key")
    voyage_api_key: str | None = Field(default=None, description="Voyage AI API key")
    claude_api_key: str | None = Field(default=None, description="Claude API key for RAG")

    # Cost controls
    max_embed_tokens_per_day: int = Field(
        default=1_000_000, description="Maximum embedding tokens per day"
    )
    max_chat_tokens_per_day: int = Field(
        default=500_000, description="Maximum chat tokens per day"
    )

    # Chunking parameters
    chunk_size_tokens: int = Field(default=800, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in tokens")

    # Indexing
    max_embed_workers: int = Field(default=4, description="Max parallel embedding workers")
    embed_batch_size: int = Field(default=64, description="Batch size for embedding")

    # Search
    semantic_weight: float = Field(default=0.7, description="Weight for semantic search")
    keyword_weight: float = Field(default=0.3, description="Weight for keyword search")
    enable_rerank: bool = Field(default=True, description="Enable cross-encoder reranking")

    # Optional: Notifications
    telegram_bot_token: str | None = Field(default=None, description="Telegram bot token")
    telegram_chat_id: str | None = Field(default=None, description="Telegram chat ID")

    # Optional: Email
    smtp_host: str | None = Field(default=None, description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_user: str | None = Field(default=None, description="SMTP username")
    smtp_password: str | None = Field(default=None, description="SMTP password")
    smtp_from: str | None = Field(default=None, description="SMTP from address")

    # Optional: Security
    redact_before_embed: bool = Field(
        default=False, description="Redact PII before embedding"
    )
    enable_encryption: bool = Field(default=False, description="Enable data encryption")
    encryption_key: str | None = Field(default=None, description="Encryption key")

    # Optional: Advanced
    enable_ocr: bool = Field(default=True, description="Enable OCR for images and scanned PDFs")
    query_expansion: bool = Field(default=True, description="Enable query expansion with LLM")
    weekly_digest_enabled: bool = Field(
        default=False, description="Enable weekly digest generation"
    )
    reflection_enabled: bool = Field(
        default=False, description="Enable reflection prompts generation"
    )

    @field_validator("solvx_data_dir", "solvx_db_path", mode="before")
    @classmethod
    def expand_paths(cls, v: str | Path) -> Path:
        """Expand and resolve paths."""
        if isinstance(v, str):
            v = Path(v)
        return v.expanduser().resolve()

    def model_post_init(self, __context) -> None:
        """Post-initialization: create directories and validate settings."""
        # Create data directories
        self.solvx_data_dir.mkdir(parents=True, exist_ok=True)
        (self.solvx_data_dir / "chroma").mkdir(exist_ok=True)
        (self.solvx_data_dir / "logs").mkdir(exist_ok=True)
        (self.solvx_data_dir / "insights").mkdir(exist_ok=True)

        # Ensure DB parent directory exists
        self.solvx_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate API keys for non-local providers
        if self.embed_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBED_PROVIDER is set to 'openai'. "
                "Please set it in your .env file or use EMBED_PROVIDER=local for free local embeddings."
            )
        elif self.embed_provider == "cohere" and not self.cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY is required when EMBED_PROVIDER is set to 'cohere'. "
                "Please set it in your .env file or use EMBED_PROVIDER=local for free local embeddings."
            )
        elif self.embed_provider == "voyage" and not self.voyage_api_key:
            raise ValueError(
                "VOYAGE_API_KEY is required when EMBED_PROVIDER is set to 'voyage'. "
                "Please set it in your .env file or use EMBED_PROVIDER=local for free local embeddings."
            )

        # Validate search weights sum to ~1.0
        total_weight = self.semantic_weight + self.keyword_weight
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"Semantic and keyword weights must sum to 1.0 (got {total_weight}). "
                "Please adjust SEMANTIC_WEIGHT and KEYWORD_WEIGHT in your .env file."
            )

    @property
    def chroma_path(self) -> Path:
        """Path to Chroma database."""
        return self.solvx_data_dir / "chroma"

    @property
    def logs_path(self) -> Path:
        """Path to logs directory."""
        return self.solvx_data_dir / "logs"

    @property
    def insights_path(self) -> Path:
        """Path to insights directory."""
        return self.solvx_data_dir / "insights"


# Global settings instance
settings = Settings()
