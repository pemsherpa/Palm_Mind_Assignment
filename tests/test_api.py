from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import app.api as api
from app.database import Base, get_db


class FakeAIService:
    def embed_texts(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def generate_answer(
        self,
        message: str,
        history: list[dict[str, str]],
        context: list[str],
    ) -> tuple[str, dict[str, str] | None]:
        return "Here is an answer.", None


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: list[tuple[str, list[str]]] = []

    def upsert_chunks(
        self,
        document_id: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        self.documents.append((document_id, chunks))

    def search(self, vector: list[float]) -> list[str]:
        return ["Relevant source."]


class FakeChatMemory:
    def __init__(self) -> None:
        self.exchanges: list[tuple[str, str, str]] = []

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        return []

    def add_exchange(self, session_id: str, user_message: str, answer: str) -> None:
        self.exchanges.append((session_id, user_message, answer))

    def get_booking_id(self, session_id: str) -> str | None:
        return None

    def save_booking_id(self, session_id: str, booking_id: str) -> None:
        pass


def create_test_client() -> tuple[TestClient, Engine]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    def get_test_db() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    test_app = FastAPI()
    test_app.include_router(api.router)
    test_app.dependency_overrides[get_db] = get_test_db
    return TestClient(test_app), engine


def test_document_ingestion_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    vector_store = FakeVectorStore()
    monkeypatch.setattr(api, "ai_service", FakeAIService())
    monkeypatch.setattr(api, "vector_store", vector_store)
    client, engine = create_test_client()

    response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"A simple document.", "text/plain")},
        data={"chunking_strategy": "fixed_size"},
    )

    assert response.status_code == 201
    assert response.json()["chunk_count"] == 1
    assert vector_store.documents[0][1] == ["A simple document."]
    client.close()
    engine.dispose()


def test_chat_endpoint_returns_answer_and_saves_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = FakeChatMemory()
    monkeypatch.setattr(api, "ai_service", FakeAIService())
    monkeypatch.setattr(api, "vector_store", FakeVectorStore())
    monkeypatch.setattr(api, "chat_memory", memory)
    client, engine = create_test_client()

    response = client.post(
        "/api/v1/chat",
        json={"session_id": "candidate-1", "message": "Tell me about the document."},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Here is an answer."
    assert memory.exchanges == [
        ("candidate-1", "Tell me about the document.", "Here is an answer.")
    ]
    client.close()
    engine.dispose()
