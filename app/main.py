from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.service import RAGService, get_rag_service


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "LangGraph + Pinecone RAG API grounded strictly in the Agentic AI eBook."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "message": "Agentic AI eBook RAG API",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(config: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=config.app_name,
        provider=config.llm_provider,
        model=config.active_llm_model,
        vector_db="Pinecone",
        embedding_model=config.embedding_model,
    )


@app.post("/chat", response_model=ChatResponse, tags=["rag"])
def chat(
    request: ChatRequest,
    service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        return service.answer(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/stats", tags=["meta"])
def vector_stats(
    service: RAGService = Depends(get_rag_service),
) -> dict:
    try:
        return service.vector_store.stats()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read Pinecone stats: {exc}",
        ) from exc
