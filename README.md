# RAG Chatbot

RAG pipeline with hybrid search, LLM re-ranking, and streaming responses. Upload PDF, TXT, or Markdown documents and query them via REST API or web interface. The FastAPI backend works independently from the Streamlit UI — designed for integration into existing systems.

No LangChain, no LlamaIndex — the entire retrieval pipeline is written from scratch in ~850 lines of Python.

![Demo](demo.png)

## Why from scratch

Most RAG tutorials boil down to `langchain.RetrievalQA` with default settings. This project implements every component manually to understand how retrieval actually works: where naive vector search fails, why hybrid search helps, and what re-ranking brings to the table.

## Architecture

```
Documents (PDF/TXT/MD)
        |
        v
  Parent-Child Chunking
  (parent: 2000 chars, child: 400 chars)
        |
        v
  Local Embeddings (all-MiniLM-L6-v2)
        |
        v
  ChromaDB (vector) + BM25 (lexical)
        |
  User Question
        |
        +---> Query Reformulation (multi-turn context)
        |
        +---> Vector Search (cosine similarity)
        +---> BM25 Search (lexical matching)
        |
        v
  Reciprocal Rank Fusion (RRF)
        |
        v
  LLM Re-ranking (relevance scoring)
        |
        v
  Parent chunk retrieval (full context)
        |
        v
  LLM Answer Generation (streaming via SSE)
        |
        v
  Answer + Sources
```

## Highlights

- **Hybrid search** — BM25 (lexical) + vector (semantic) with Reciprocal Rank Fusion
- **LLM re-ranking** — relevance scoring of retrieved chunks before answer generation
- **Parent-child chunking** — small chunks for precise retrieval, large chunks for rich LLM context
- **Streaming** — Server-Sent Events for real-time token-by-token response
- **Multi-turn** — query reformulation using conversation history
- **Chat history** — SQLite-backed persistent chat sessions
- **Multi-provider** — switch between Groq (free) and OpenAI from the UI
- **Local embeddings** — all-MiniLM-L6-v2 via ONNX, no external API needed
- **No framework overhead** — vanilla Python + direct API calls, no LangChain

## Design decisions

**Parent-child chunking vs fixed-size chunks.** Small chunks (400 chars) give precise retrieval — the embedding matches exactly what the user asked. But feeding a 400-char snippet to the LLM loses surrounding context. Parent chunks (2000 chars) solve this: search on children, answer from parents.

**Hybrid search vs pure vector.** Semantic search is great for meaning but bad for exact terms — names, dates, error codes, abbreviations. BM25 catches these. RRF merges both ranked lists without needing to normalize scores.

**LLM re-ranking vs cross-encoder.** A cross-encoder would be faster and cheaper, but requires a separate model. LLM re-ranking reuses the same model that generates answers, keeps the stack simple, and works well enough for the candidate set size (~40 chunks).

**No LangChain.** Full control over prompts, chunking strategy, and retrieval logic. Easier to debug, fewer dependencies, transparent behavior.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (Llama 3.3 70B, Llama 4, Qwen 3) / OpenAI (GPT-4.1) |
| Embeddings | all-MiniLM-L6-v2 (local, ONNX) |
| Vector DB | ChromaDB |
| Lexical search | BM25 (rank-bm25) |
| API | FastAPI + SSE streaming |
| UI | Streamlit |
| Chat storage | SQLite |
| Container | Docker |

## Quick Start

### With Docker

```bash
cp .env.example .env
# Add your API keys to .env (Groq is free, OpenAI is optional)

docker compose up --build
```

### Without Docker

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env (Groq is free, OpenAI is optional)

# Terminal 1 — API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — UI
streamlit run app/ui.py --server.port 8501 --server.headless true
```

Open http://localhost:8501

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Upload and index a document (PDF/TXT/MD) |
| POST | `/query` | Ask a question (supports SSE streaming) |
| GET | `/providers` | List available LLM providers and models |
| POST | `/model` | Switch LLM provider/model at runtime |
| GET | `/stats` | Chunk counts |
| GET | `/health` | Health check |

### Examples

```bash
# Upload a document
curl -X POST http://localhost:8000/ingest -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "stream": false}'

# Switch model
curl -X POST http://localhost:8000/model \
  -H "Content-Type: application/json" \
  -d '{"provider": "groq", "model": "llama-3.1-8b-instant"}'
```

## Limitations

- Parent chunk store is in-memory — lost on restart (needs persistent storage)
- No deduplication on re-ingesting the same file
- BM25 index rebuilds fully on each ingest
- No API authentication

## Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py            # FastAPI endpoints
│   ├── rag.py             # RAG pipeline (ingest, search, query)
│   ├── chat_history.py    # SQLite chat persistence
│   ├── config.py          # Configuration and provider settings
│   └── ui.py              # Streamlit web interface
├── documents/             # Uploaded documents (gitignored)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```
