from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.database import create_database_tables


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    create_database_tables()
    yield


app = FastAPI(
    title="Palm Mind RAG API",
    description="Document ingestion and conversational retrieval-augmented generation.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
