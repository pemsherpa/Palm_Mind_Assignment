from io import BytesIO

from pypdf import PdfReader


def extract_text(filename: str, content: bytes) -> str:
    if filename.lower().endswith(".txt"):
        return content.decode("utf-8-sig").strip()
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    raise ValueError("Only .pdf and .txt files are supported.")
