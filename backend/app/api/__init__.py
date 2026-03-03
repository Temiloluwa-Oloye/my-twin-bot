from fastapi import APIRouter

from .routes import router as chat_router

router = APIRouter()
router.include_router(chat_router, tags=["chat"])

