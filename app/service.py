from functools import lru_cache

from app.config import get_settings
from app.embeddings import EmbeddingModel
from app.graph import RAGGraph
from app.llm import LLMClient
from app.schemas import ChatResponse, ContextChunk
from app.vector_store import PineconeVectorStore


class RAGService:
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = EmbeddingModel(self.settings.embedding_model)
        self.vector_store = PineconeVectorStore(
            api_key=self.settings.pinecone_api_key or "",
            index_name=self.settings.pinecone_index_name,
            namespace=self.settings.pinecone_namespace,
            cloud=self.settings.pinecone_cloud,
            region=self.settings.pinecone_region,
            embeddings=self.embeddings,
        )
        self.llm = LLMClient(self.settings)
        self.rag_graph = RAGGraph(
            settings=self.settings,
            vector_store=self.vector_store,
            llm=self.llm,
        )

    def answer(self, question: str) -> ChatResponse:
        state = self.rag_graph.invoke(question)

        contexts = [
            ContextChunk(
                chunk_id=item["chunk_id"],
                page=item.get("page"),
                score=float(item.get("score", 0.0)),
                content=item.get("content", ""),
            )
            for item in state.get("retrieved_contexts", [])
        ]

        return ChatResponse(
            question=state["question"],
            answer=state["answer"],
            confidence=float(state.get("confidence", 0.0)),
            grounded=bool(state.get("grounded", False)),
            context_chunks=contexts,
            provider=self.settings.llm_provider,
            model=self.settings.active_llm_model,
        )


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService()
