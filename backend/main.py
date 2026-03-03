from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.retrieval.embedder import LocalEmbedder
from app.retrieval.vector_store import ChromaVectorStore
from app.services.llm_service import LLMService
from app.services.memory import InMemorySessionMemory


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="My Twin Bot API",
        version="0.1.0",
        description="Backend for Temi's Digital Twin chatbot.",
    )

    # CORS configuration – adjust for your frontend origin in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Core shared components
    embedder = LocalEmbedder(model_name=settings.embedding_model_name)
    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_db_dir,
        collection_name=settings.chroma_collection,
    )
    memory = InMemorySessionMemory()
    llm_service = LLMService(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model_name,
    )

    # Attach to app state for access inside routes
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.memory = memory
    app.state.llm_service = llm_service
    app.state.settings = settings

    app.include_router(api_router, prefix="/api")

    return app


app = create_app()


@app.get("/health", tags=["health"])
async def root_health() -> dict:
    return {"status": "ok"}

