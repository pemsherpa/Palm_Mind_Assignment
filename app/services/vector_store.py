from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.config import Settings


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
        self._collection = settings.qdrant_collection

    def _ensure_collection(self, vector_size: int) -> None:
        collections = self._client.get_collections().collections
        if any(collection.name == self._collection for collection in collections):
            info = self._client.get_collection(self._collection)
            configured_size = info.config.params.vectors.size
            if configured_size != vector_size:
                raise RuntimeError(
                    f"Qdrant collection has vector size {configured_size}, expected {vector_size}."
                )
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert_chunks(
        self,
        document_id: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        if not vectors:
            return
        if len(chunks) != len(vectors):
            raise ValueError("Each document chunk must have exactly one embedding.")
        self._ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{document_id}:{index}")),
                vector=vector,
                payload={"document_id": document_id, "chunk_index": index, "text": chunk},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int = 5) -> list[str]:
        self._ensure_collection(len(vector))
        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            point.payload["text"]
            for point in results.points
            if point.payload and isinstance(point.payload.get("text"), str)
        ]
