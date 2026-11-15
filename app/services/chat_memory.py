import json

from redis import Redis

from app.config import Settings


class ChatMemory:
    def __init__(self, settings: Settings) -> None:
        self._client: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self._history_turns = settings.chat_history_turns
        self._ttl_seconds = settings.chat_memory_ttl_seconds

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        raw_messages = self._client.lrange(f"chat:{session_id}", 0, -1)
        return [json.loads(message) for message in raw_messages]

    def add_exchange(self, session_id: str, user_message: str, answer: str) -> None:
        key = f"chat:{session_id}"
        self._client.rpush(
            key,
            json.dumps({"role": "user", "content": user_message}),
            json.dumps({"role": "assistant", "content": answer}),
        )
        self._client.ltrim(key, -self._history_turns * 2, -1)
        self._client.expire(key, self._ttl_seconds)

    def get_booking_id(self, session_id: str) -> str | None:
        return self._client.get(f"chat:{session_id}:booking")

    def save_booking_id(self, session_id: str, booking_id: str) -> None:
        self._client.set(
            f"chat:{session_id}:booking",
            booking_id,
            ex=self._ttl_seconds,
        )
