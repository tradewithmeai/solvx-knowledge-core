# SolVX Knowledge Core

Local-first personal knowledge base with RAG, hybrid search, and intelligent insights.

## Features

- **Multi-format parsing**: PDF, Markdown, HTML, DOCX, CSV, Jupyter notebooks, code files, and more
- **Hybrid search**: Combines keyword (FTS5) and semantic (vector) search with optional cross-encoder reranking
- **RAG with Claude**: Streaming chat with citations and source anchors
- **Local-first**: Runs entirely on your machine; choose local embeddings (free) or API providers
- **Memory graph**: Visualize connections between documents
- **Weekly digests**: Automatic summaries of new knowledge
- **Reflection prompts**: AI-generated questions based on your knowledge trends
- **Privacy-focused**: Optional PII detection and redaction
- **Cost controls**: Daily token limits and spend tracking

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- (Optional) Tesseract for OCR

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd solvx-knowledge-core

# Install dependencies
make install

# Copy and configure environment
cp .env.example .env
# Edit .env to set your preferences (default uses free local embeddings)

# Initialize database
make upgrade

# Run the server
make dev
```

The API will be available at `http://127.0.0.1:8787`

### Using Local Embeddings (Free)

By default, SolVX uses local sentence transformers for embeddings (no API key required):

```bash
# In .env
EMBED_PROVIDER=local
```

### Using Cloud Embeddings (More Accurate)

For better accuracy, use OpenAI, Cohere, or Voyage:

```bash
# In .env
EMBED_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

### Claude API for Chat

To use the RAG chat feature, you'll need a Claude API key:

```bash
# In .env
CLAUDE_API_KEY=your_anthropic_key_here
```

## Architecture

- **Backend**: Python 3.11, FastAPI, SQLAlchemy, Alembic
- **Vector Store**: Chroma (persistent local storage)
- **Keyword Search**: SQLite FTS5
- **Embeddings**: Sentence Transformers (local) or OpenAI/Cohere/Voyage (API)
- **LLM**: Claude via Anthropic API
- **Frontend**: React + Vite + TypeScript + TailwindCSS
- **Graph**: D3.js force-layout visualization

## Development

```bash
# Run tests
make test

# Format code
make fmt

# Lint code
make lint

# Create migration
make migrate

# Apply migrations
make upgrade
```

## CLI Usage

```bash
# Add a directory to watch
poetry run solvx scan add ~/Documents

# Run scan
poetry run solvx scan run

# Search
poetry run solvx search "quantum physics"

# Re-embed all documents
poetry run solvx index reembed --all

# Generate weekly digest
poetry run solvx digest weekly
```

## API Endpoints

- `GET /api/health` - Health check and statistics
- `POST /api/paths` - Add watch path
- `POST /api/scan` - Trigger scan
- `GET /api/search` - Hybrid search
- `POST /api/chat` - RAG chat (SSE streaming)
- `GET /api/graph` - Memory graph
- `GET /api/insights/weekly/latest` - Latest weekly digest

See `/docs` for full API documentation (OpenAPI).

## Configuration

All configuration is done via environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBED_PROVIDER` | Embedding provider (local, openai, cohere, voyage) | `local` |
| `CHUNK_SIZE_TOKENS` | Target chunk size | `800` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `SEMANTIC_WEIGHT` | Weight for semantic search | `0.7` |
| `KEYWORD_WEIGHT` | Weight for keyword search | `0.3` |
| `ENABLE_RERANK` | Enable cross-encoder reranking | `true` |
| `MAX_EMBED_TOKENS_PER_DAY` | Daily embedding token limit | `1000000` |
| `MAX_CHAT_TOKENS_PER_DAY` | Daily chat token limit | `500000` |

See `.env.example` for all options.

## Security & Privacy

- Runs on `localhost` only by default
- Optional PII detection and redaction
- API keys stored in `.env` (never logged)
- Encryption at rest available (SQLCipher)
- Respects `.solvxignore` for excluded paths

## Project Structure

```
solvx-knowledge-core/
├── backend/
│   ├── app/
│   │   ├── config.py          # Configuration
│   │   ├── main.py            # FastAPI app
│   │   ├── logging.py         # Structured logging
│   │   ├── models/            # SQLAlchemy models
│   │   ├── db/                # Database session & migrations
│   │   ├── ingestion/         # Parsers, scanner, chunking
│   │   ├── embeddings/        # Embedding providers & cache
│   │   ├── vectorstore/       # Chroma integration
│   │   ├── search/            # Hybrid search & reranking
│   │   ├── rag/               # RAG assembly & Claude client
│   │   ├── scheduler/         # Background jobs
│   │   ├── insights/          # Weekly digest & reflection
│   │   ├── graph/             # Graph builder
│   │   ├── routes/            # API routes
│   │   └── utils/             # Utilities
├── cli/                       # CLI tool
├── frontend/                  # React web UI
├── tests/                     # Tests
├── data/                      # Local data (gitignored)
├── pyproject.toml            # Python dependencies
├── Makefile                  # Development commands
└── README.md                 # This file
```

## License

MIT

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Support

For issues and feature requests, please use GitHub Issues.
