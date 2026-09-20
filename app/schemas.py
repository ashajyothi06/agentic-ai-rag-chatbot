from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class ContextChunk(BaseModel):
    chunk_id: str
    page: int | None = None
    score: float
    content: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    grounded: bool
    context_chunks: list[ContextChunk]
    provider: str
    model: str


class HealthResponse(BaseModel):
    status: str
    app: str
    provider: str
    model: str
    vector_db: str
    embedding_model: str


class IngestionResponse(BaseModel):
    pages: int
    chunks: int
    index_name: str
    namespace: str
