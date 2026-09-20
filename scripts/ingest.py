import argparse

from app.config import get_settings
from app.embeddings import EmbeddingModel
from app.pdf_loader import load_and_chunk_pdf
from app.vector_store import PineconeVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk the Agentic AI PDF, embed it, and store vectors in Pinecone."
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Optional PDF path. Defaults to SOURCE_PDF_PATH from .env.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the configured namespace before ingesting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()

    pdf_path = args.pdf or settings.source_pdf_path

    print(f"Loading PDF: {pdf_path}")
    chunks, pages = load_and_chunk_pdf(
        pdf_path=pdf_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print(f"Extracted {len(chunks)} chunks from {pages} PDF pages.")
    print(f"Embedding model: {settings.embedding_model}")

    embeddings = EmbeddingModel(settings.embedding_model)
    store = PineconeVectorStore(
        api_key=settings.pinecone_api_key or "",
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        embeddings=embeddings,
    )

    store.ensure_index()

    if args.reset:
        print(f"Clearing namespace: {settings.pinecone_namespace}")
        store.clear_namespace()

    count = store.upsert_chunks(chunks)
    print(
        f"Done. Upserted {count} chunks into index "
        f"'{settings.pinecone_index_name}' / namespace "
        f"'{settings.pinecone_namespace}'."
    )


if __name__ == "__main__":
    main()
