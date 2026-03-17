import threading
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.retrieval.embedder import LocalEmbedder
from app.retrieval.vector_store import ChromaVectorStore
from app.services.llm_service import LLMService
from app.services.memory import InMemorySessionMemory

# A thread lock to ensure only the first request triggers the model download
model_lock = threading.Lock()

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="My Twin Bot API",
        version="0.1.0",
        description="Backend for Temi's Digital Twin chatbot.",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1. Initialize fast components only
    app.state.settings = settings
    app.state.memory = InMemorySessionMemory()
    
    # 2. Leave heavy ML components empty for now (Lazy Loading)
    app.state.embedder = None
    app.state.vector_store = None
    app.state.llm_service = None
    app.state.models_loaded = False

    app.include_router(api_router, prefix="/api")

    return app

app = create_app()

# 3. The Lazy Loading Middleware
@app.middleware("http")
async def lazy_load_components(request: Request, call_next):
    # Check if this is the very first request
    if not getattr(request.app.state, "models_loaded", False):
        with model_lock:
            # Double-check inside the lock to prevent race conditions
            if not getattr(request.app.state, "models_loaded", False):
                print("🚀 First request received! Waking up AI models... (This takes 1-2 mins)")
                settings = request.app.state.settings
                
                # Now we safely load the heavy models in the background
                request.app.state.embedder = LocalEmbedder(model_name=settings.embedding_model_name)
                request.app.state.vector_store = ChromaVectorStore(
                    persist_directory=settings.chroma_db_dir,
                    collection_name=settings.chroma_collection,
                )
                request.app.state.llm_service = LLMService(
                    api_key=settings.groq_api_key,
                    model_name=settings.groq_model_name,
                )
                
                request.app.state.models_loaded = True
                print("✅ AI models loaded successfully!")
    
    response = await call_next(request)
    return response

@app.get("/health", tags=["health"])
async def root_health() -> dict:
    return {
        "status": "ok", 
        "models_loaded": getattr(app.state, "models_loaded", False)
    }