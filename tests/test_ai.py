import json
from collections.abc import Callable

import httpx
import pytest

from app.config import Settings
from app.services.ai import AIService


def create_ai_service(
    handler: Callable[[httpx.Request], httpx.Response],
    api_key: str = "test-key",
) -> AIService:
    client = httpx.Client(
        base_url="https://api.cohere.com/v2",
        headers={"Authorization": f"Bearer {api_key}"},
        transport=httpx.MockTransport(handler),
    )
    return AIService(Settings(cohere_api_key=api_key), client=client)


def test_embeddings_use_cohere_search_document_input() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={"embeddings": {"float": [[0.1, 0.2] for _ in payload["texts"]]}},
        )

    ai_service = create_ai_service(handler)

    vectors = ai_service.embed_texts(["first", "second"], input_type="search_document")

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert requests == [
        {
            "model": "embed-v4.0",
            "texts": ["first", "second"],
            "input_type": "search_document",
            "embedding_types": ["float"],
        }
    ]


def test_chat_parses_cohere_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "command-a-03-2025"
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "answer": "Hello.",
                                    "booking_intent": True,
                                    "booking": {"name": "Alex"},
                                }
                            ),
                        }
                    ]
                }
            },
        )

    ai_service = create_ai_service(handler)

    answer, booking = ai_service.generate_answer("Hello", [], [])

    assert answer == "Hello."
    assert booking == {"name": "Alex"}


def test_ai_service_requires_cohere_api_key() -> None:
    ai_service = AIService(Settings(cohere_api_key=""))

    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        ai_service.embed_texts(["text"], input_type="search_query")
