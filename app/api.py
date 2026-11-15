from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pypdf.errors import PdfReadError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Document, DocumentChunk
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkingStrategy,
    DocumentIngestionResponse,
)
from app.services.ai import AIService
from app.services.booking import create_booking
from app.services.chat_memory import ChatMemory
from app.services.chunking import fixed_size_chunks, sentence_chunks
from app.services.document_reader import extract_text
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/v1")
settings = get_settings()
ai_service = AIService(settings)
vector_store = VectorStore(settings)
chat_memory = ChatMemory(settings)


@router.post(
    "/documents",
    response_model=DocumentIngestionResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_document(
    file: Annotated[UploadFile, File()],
    chunking_strategy: Annotated[ChunkingStrategy, Form()],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentIngestionResponse:
    filename = file.filename or ""
    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=415, detail="Only .pdf and .txt files are supported.")

    content = file.file.read(settings.max_upload_size_bytes + 1)
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="The uploaded file exceeds the size limit.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        text = extract_text(filename, content)
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (PdfReadError, OSError) as error:
        raise HTTPException(
            status_code=400, detail="The uploaded PDF could not be read."
        ) from error
    if not text:
        raise HTTPException(status_code=400, detail="No readable text was found in the document.")

    chunks = (
        fixed_size_chunks(text)
        if chunking_strategy == ChunkingStrategy.FIXED_SIZE
        else sentence_chunks(text)
    )
    document = Document(
        filename=filename[:255],
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(content),
        chunking_strategy=chunking_strategy.value,
        chunk_count=len(chunks),
    )
    db.add(document)
    db.flush()

    try:
        vectors = ai_service.embed_texts(chunks, input_type="search_document")
        vector_store.upsert_chunks(document.id, chunks, vectors)
        db.add_all(
            [
                DocumentChunk(document_id=document.id, chunk_index=index, text=chunk)
                for index, chunk in enumerate(chunks)
            ]
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    return DocumentIngestionResponse(
        document_id=document.id,
        filename=document.filename,
        chunking_strategy=chunking_strategy,
        chunk_count=document.chunk_count,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
) -> ChatResponse:
    history = chat_memory.get_history(request.session_id)
    query_vector = ai_service.embed_texts(
        [request.message], input_type="search_query"
    )[0]
    context = vector_store.search(query_vector)
    answer, booking_data = ai_service.generate_answer(request.message, history, context)

    booking_id: str | None = None
    if booking_data and not chat_memory.get_booking_id(request.session_id):
        booking, booking_prompt = create_booking(db, booking_data)
        if booking_prompt:
            answer = booking_prompt
        elif booking:
            booking_id = booking.id
            chat_memory.save_booking_id(request.session_id, booking.id)
            answer = (
                f"Your interview is booked for {booking.interview_date.isoformat()} at "
                f"{booking.interview_time.strftime('%H:%M')}."
            )

    chat_memory.add_exchange(request.session_id, request.message, answer)
    return ChatResponse(session_id=request.session_id, answer=answer, booking_id=booking_id)
