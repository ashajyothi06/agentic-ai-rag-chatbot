import time
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec
from pinecone.errors.exceptions import NotFoundError

from app.embeddings import EmbeddingModel
from app.pdf_loader import ChunkRecord


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    text: str
    page: int | None
    score: float
    source: str | None = None


class PineconeVectorStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str,
        cloud: str,
        region: str,
        embeddings: EmbeddingModel,
    ):
        if not api_key:
            raise ValueError("PINECONE_API_KEY is required.")

        self.index_name = index_name
        self.namespace = namespace
        self.cloud = cloud
        self.region = region
        self.embeddings = embeddings
        self.pc = Pinecone(api_key=api_key)
        self._index = None

    def ensure_index(self) -> None:
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embeddings.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

            deadline = time.time() + 90
            while time.time() < deadline:
                description = self.pc.describe_index(self.index_name)
                status = getattr(description, "status", None)

                ready = False
                if isinstance(status, dict):
                    ready = bool(status.get("ready"))
                elif status is not None:
                    ready = bool(getattr(status, "ready", False))

                if ready:
                    break
                time.sleep(2)

        description = self.pc.describe_index(self.index_name)
        existing_dimension = getattr(description, "dimension", None)
        if existing_dimension is None and isinstance(description, dict):
            existing_dimension = description.get("dimension")

        if existing_dimension and int(existing_dimension) != self.embeddings.dimension:
            raise ValueError(
                f"Pinecone index '{self.index_name}' has dimension "
                f"{existing_dimension}, but embedding model '{self.embeddings.model_name}' "
                f"uses {self.embeddings.dimension}. Delete/recreate the index or change "
                "PINECONE_INDEX_NAME."
            )

        self._index = self.pc.Index(self.index_name)

    @property
    def index(self):
        if self._index is None:
            self.ensure_index()
        return self._index

    def clear_namespace(self) -> None:
        """Delete all vectors in the namespace.

        A namespace is created lazily by Pinecone on first upsert. Therefore,
        the first ever `--reset` can legitimately receive HTTP 404 because
        there is nothing to delete yet. Treat that specific case as success.
        """
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            print(f"Cleared Pinecone namespace: {self.namespace}")
        except NotFoundError:
            print(
                f"Pinecone namespace '{self.namespace}' does not exist yet; "
                "nothing to clear."
            )

    def upsert_chunks(self, chunks: list[ChunkRecord], batch_size: int = 100) -> int:
        total = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embeddings.embed_documents([item.text for item in batch])

            records = []
            for item, vector in zip(batch, vectors, strict=True):
                records.append(
                    {
                        "id": item.chunk_id,
                        "values": vector,
                        "metadata": {
                            "text": item.text,
                            "page": item.page,
                            "source": item.source,
                        },
                    }
                )

            self.index.upsert(vectors=records, namespace=self.namespace)
            total += len(records)

        return total

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_vector = self.embeddings.embed_query(query)

        result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        matches = getattr(result, "matches", None)
        if matches is None and isinstance(result, dict):
            matches = result.get("matches", [])

        hits: list[SearchHit] = []
        for match in matches or []:
            if isinstance(match, dict):
                match_id = str(match.get("id", ""))
                score = float(match.get("score", 0.0))
                metadata = match.get("metadata") or {}
            else:
                match_id = str(getattr(match, "id", ""))
                score = float(getattr(match, "score", 0.0))
                metadata = getattr(match, "metadata", None) or {}

            hits.append(
                SearchHit(
                    chunk_id=match_id,
                    text=str(metadata.get("text", "")),
                    page=_safe_int(metadata.get("page")),
                    score=score,
                    source=metadata.get("source"),
                )
            )

        return hits

    def stats(self) -> dict:
        response = self.index.describe_index_stats()
        if hasattr(response, "to_dict"):
            return response.to_dict()
        if isinstance(response, dict):
            return response
        return {"raw": str(response)}


def _safe_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
