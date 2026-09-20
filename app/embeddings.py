from functools import cached_property

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Local text embeddings so the RAG pipeline does not need a second paid API."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise ValueError(
                f"Could not determine embedding dimension for {self.model_name}."
            )
        return int(dimension)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()
