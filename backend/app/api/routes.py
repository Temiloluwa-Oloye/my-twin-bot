from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class RetrievedContextChunk(BaseModel):
    id: str
    text: str
    source: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    context: List[RetrievedContextChunk]


@router.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request) -> ChatResponse:
    app_state = request.app.state

    vector_store = app_state.vector_store
    embedder = app_state.embedder
    memory = app_state.memory
    llm_service = app_state.llm_service

    # Retrieve conversation history for this session
    history = memory.get_history(payload.session_id)

    # Embed user message and query vector store
    query_embedding = embedder.embed_query(payload.message)
    retrieved_docs = vector_store.query_by_embedding(
        query_embedding=query_embedding,
        top_k=3,
    )

    if not retrieved_docs:
        raise HTTPException(
            status_code=500,
            detail="No knowledge base context available. Please run ingestion first.",
        )

    # Ask LLM for an answer grounded in retrieved context and history
    answer = await llm_service.generate_reply(
        user_message=payload.message,
        context_documents=retrieved_docs,
        history=history,
        session_id=payload.session_id,
    )

    # Update memory with this turn
    memory.append_message(
        session_id=payload.session_id,
        role="user",
        content=payload.message,
    )
    memory.append_message(
        session_id=payload.session_id,
        role="assistant",
        content=answer,
    )

    context_chunks = [
        RetrievedContextChunk(
            id=doc.id,
            text=doc.text,
            source=doc.metadata.get("source") if doc.metadata else None,
        )
        for doc in retrieved_docs
    ]

    return ChatResponse(answer=answer, context=context_chunks)

