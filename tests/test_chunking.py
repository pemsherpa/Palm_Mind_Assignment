import pytest

from app.services.chunking import fixed_size_chunks, sentence_chunks


def test_fixed_size_chunks_overlap_without_splitting_words() -> None:
    text = "one two three four five six seven eight"

    chunks = fixed_size_chunks(text, chunk_size=20, overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert all(set(chunk.split()).issubset(set(text.split())) for chunk in chunks)
    reconstructed: list[str] = []
    for chunk in chunks:
        words = chunk.split()
        overlap = max(
            (
                size
                for size in range(1, min(len(reconstructed), len(words)) + 1)
                if reconstructed[-size:] == words[:size]
            ),
            default=0,
        )
        reconstructed.extend(words[overlap:])
    assert reconstructed == text.split()


def test_sentence_chunks_keep_sentences_together() -> None:
    chunks = sentence_chunks("First sentence. Second sentence. Third sentence.", max_chunk_size=32)

    assert chunks == ["First sentence. Second sentence.", "Third sentence."]


def test_chunkers_return_empty_for_blank_input() -> None:
    assert fixed_size_chunks("  \n\t ") == []
    assert sentence_chunks("") == []


def test_fixed_size_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        fixed_size_chunks("text", chunk_size=10, overlap=10)
