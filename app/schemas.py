from enum import StrEnum

from pydantic import BaseModel, Field


class ChunkingStrategy(StrEnum):
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"


class DocumentIngestionResponse(BaseModel):
    document_id: str
    filename: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    booking_id: str | None = None
