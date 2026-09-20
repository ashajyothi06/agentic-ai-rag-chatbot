from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Agentic AI eBook RAG"
    environment: str = "development"

    # LLM provider: qwen or grok
    llm_provider: Literal["qwen", "grok"] = "qwen"

    qwen_api_key: str | None = None
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    xai_api_key: str | None = None
    xai_base_url: str = "https://api.x.ai/v1"
    xai_model: str = "grok-4.6"

    # Pinecone
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "agentic-ai-ebook"
    pinecone_namespace: str = "ebook-v1"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Embeddings / chunking
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(default=900, ge=200, le=3000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)

    # Retrieval / grounding
    top_k: int = Field(default=5, ge=1, le=12)
    min_similarity: float = Field(default=0.25, ge=-1.0, le=1.0)
    min_grounding_score: float = Field(default=0.75, ge=0.0, le=1.0)
    max_generation_attempts: int = Field(default=2, ge=1, le=3)

    # Source document
    source_pdf_url: str = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    source_pdf_path: str = "data/Ebook-Agentic-AI.pdf"

    # API
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    @property
    def active_llm_api_key(self) -> str:
        if self.llm_provider == "qwen":
            if not self.qwen_api_key:
                raise ValueError("QWEN_API_KEY is required when LLM_PROVIDER=qwen.")
            return self.qwen_api_key

        if not self.xai_api_key:
            raise ValueError("XAI_API_KEY is required when LLM_PROVIDER=grok.")
        return self.xai_api_key

    @property
    def active_llm_base_url(self) -> str:
        return self.qwen_base_url if self.llm_provider == "qwen" else self.xai_base_url

    @property
    def active_llm_model(self) -> str:
        return self.qwen_model if self.llm_provider == "qwen" else self.xai_model

    @property
    def parsed_cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
