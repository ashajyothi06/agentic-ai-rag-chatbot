import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    text: str
    page: int
    source: str


def load_and_chunk_pdf(
    pdf_path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[ChunkRecord], int]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found at {path}. Run: python -m scripts.download_pdf"
        )

    reader = PdfReader(str(path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[ChunkRecord] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        page_chunks = splitter.split_text(text)

        for position, chunk_text in enumerate(page_chunks):
            cleaned = " ".join(chunk_text.split()).strip()
            if not cleaned:
                continue

            raw_id = f"{path.name}:{page_number}:{position}:{cleaned[:100]}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:20]

            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=cleaned,
                    page=page_number,
                    source=path.name,
                )
            )

    if not chunks:
        raise ValueError("No extractable text was found in the PDF.")

    return chunks, len(reader.pages)
