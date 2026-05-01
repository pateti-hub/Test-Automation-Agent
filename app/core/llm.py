from __future__ import annotations

from langchain_groq import ChatGroq

from app.config import settings


def create_llm() -> ChatGroq:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0.2,
    )
