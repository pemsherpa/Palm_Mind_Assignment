# Palm Mind RAG API

A modular FastAPI backend for document ingestion and conversational RAG. It uses Cohere for embeddings and chat completion, Qdrant for vector search, Redis for per-session conversation memory, and SQLAlchemy for document and interview-booking metadata.

## Features

- Upload text-based PDF or UTF-8 TXT documents.
- Select `fixed_size` or `sentence` chunking during ingestion.
- Store embedded chunks in Qdrant and document/chunk metadata in SQL.
- Retrieve relevant content for multi-turn chat using Redis history.
- Extract and validate interview booking details from conversation; save completed bookings in SQL.
- OpenAPI documentation at `/docs`.

## Run locally

Requirements: Python 3.11+, Docker, and a Cohere API key.

1. Copy `.env.example` to `.env` and set `COHERE_API_KEY` from your Cohere dashboard.
2. Start the supporting services: `docker compose up -d qdrant redis`
3. Install and start the API:

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   uvicorn app.main:app --reload
   ```

Alternatively, run the complete stack with `docker compose up --build`; the API is available on port 8000. Set `COHERE_API_KEY` in the shell or `.env` before starting.

## API

### Ingest a document

`POST /api/v1/documents` as multipart form data:

- `file`: `.pdf` or `.txt`
- `chunking_strategy`: `fixed_size` or `sentence`

### Chat and book an interview

`POST /api/v1/chat` with JSON:

```json
{
  "session_id": "candidate-session",
  "message": "I would like to book an interview next Tuesday at 14:30. My name is Alex Doe and my email is alex@example.com."
}
```

The same `session_id` continues the conversation. Booking details can be supplied over multiple turns. The API asks for missing or invalid fields and stores a booking only when all fields are valid.

## Configuration

All settings can be supplied as environment variables. See `.env.example` for defaults. Cohere uses `command-a-03-2025` for chat and `embed-v4.0` for embeddings by default; both can be changed independently. `MAX_UPLOAD_SIZE_BYTES` defaults to 10 MiB. Redis conversation history is bounded by `CHAT_HISTORY_TURNS` and expires after `CHAT_MEMORY_TTL_SECONDS`.

## Tests

```sh
pip install -e ".[dev]"
pytest
ruff check .
```
# Palm_Mind_Assignment
