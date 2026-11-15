import re


def fixed_size_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size.")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        next_start = max(end - overlap, start + 1)
        boundary = normalized.rfind(" ", start, next_start)
        start = boundary + 1 if boundary >= start else next_start
    return [chunk for chunk in chunks if chunk]


def sentence_chunks(text: str, max_chunk_size: int = 1000) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError("Maximum chunk size must be positive.")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(fixed_size_chunks(sentence, max_chunk_size, 0))
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chunk_size:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks
