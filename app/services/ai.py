import json
from typing import Any, Literal

import httpx

from app.config import Settings


class AIService:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._client = client or (
            httpx.Client(
                base_url="https://api.cohere.com/v2",
                headers={"Authorization": f"Bearer {settings.cohere_api_key}"},
                timeout=60.0,
            )
            if settings.cohere_api_key
            else None
        )

    def _require_client(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError("COHERE_API_KEY must be configured to use document and chat APIs.")
        return self._client

    def embed_texts(
        self,
        texts: list[str],
        input_type: Literal["search_document", "search_query"],
    ) -> list[list[float]]:
        if not texts:
            return []
        client = self._require_client()
        vectors: list[list[float]] = []
        batch_size = 96
        for start in range(0, len(texts), batch_size):
            response = client.post(
                "/embed",
                json={
                    "model": self._settings.cohere_embedding_model,
                    "texts": texts[start : start + batch_size],
                    "input_type": input_type,
                    "embedding_types": ["float"],
                },
            )
            response.raise_for_status()
            payload = response.json()
            embedding_payload = payload.get("embeddings")
            embeddings = (
                embedding_payload.get("float")
                if isinstance(embedding_payload, dict)
                else None
            )
            if not isinstance(embeddings, list) or len(embeddings) != len(
                texts[start : start + batch_size]
            ):
                raise RuntimeError("The Cohere embedding response had an invalid shape.")
            if any(
                not isinstance(vector, list)
                or not vector
                or any(not isinstance(value, (int, float)) for value in vector)
                for vector in embeddings
            ):
                raise RuntimeError("The Cohere embedding response contained invalid vectors.")
            vectors.extend(embeddings)
        return vectors

    def generate_answer(
        self,
        message: str,
        history: list[dict[str, str]],
        context: list[str],
    ) -> tuple[str, dict[str, Any] | None]:
        client = self._require_client()
        context_text = "\n\n".join(context) if context else "No relevant documents were found."
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer using the supplied document context when "
                    "relevant; do not claim facts absent from it. The user may also book an "
                    "interview. Return only a JSON object with keys `answer`, `booking_intent`, "
                    "and `booking`. `booking_intent` is true only when the user is trying to book "
                    "an interview. `booking` must contain name, email, date (YYYY-MM-DD), and "
                    "time (24-hour HH:MM) found anywhere in the conversation, using null for "
                    "unknown fields. Never invent booking details. Keep `answer` concise. "
                    f"\n\nDocument context:\n{context_text}"
                ),
            },
            *history,
            {"role": "user", "content": message},
        ]
        response = client.post(
            "/chat",
            json={
                "model": self._settings.cohere_chat_model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        payload = response.json()
        message_payload = payload.get("message")
        content = (
            message_payload.get("content", [])
            if isinstance(message_payload, dict)
            else []
        )
        raw_response = "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
        if not raw_response.strip():
            raise RuntimeError("The AI service returned an empty response.")
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise RuntimeError("The AI service returned an invalid response.") from error
        if not isinstance(result, dict):
            raise RuntimeError("The AI service response must be a JSON object.")
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The AI service response did not include a valid answer.")
        booking = result.get("booking") if result.get("booking_intent") is True else None
        return answer.strip(), booking if isinstance(booking, dict) else None
