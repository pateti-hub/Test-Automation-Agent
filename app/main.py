from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.agents.irata_agent import build_irata_graph
from app.config import settings
from app.core.llm import create_llm
from app.core.loaders import SUPPORTED_EXTENSIONS, load_documents_from_path
from app.core.rag import RAGStore
from app.schemas import IngestRequest, QueryRequest, QueryResponse

app = FastAPI(title="IRATA")
store = RAGStore(settings.embedding_model)

try:
    llm = create_llm()
except RuntimeError:
    llm = None

agent = build_irata_graph(llm, store)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "llm_configured": bool(settings.groq_api_key)}


@app.post("/ingest")
async def ingest(request: IngestRequest) -> dict:
    chunks = store.ingest(request.text, request.source)
    return {"chunks": chunks}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        docs = load_documents_from_path(tmp_path, source=file.filename or tmp_path.name)
        chunks = store.ingest_documents(docs)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"chunks": chunks}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not store.query(request.question):
        raise HTTPException(status_code=404, detail="No requirements found")
    result = agent.invoke({
        "question": request.question,
        "framework": request.framework,
    })
    return QueryResponse(
        requirement_summary=result.get("requirement_summary", ""),
        testcases=result.get("testcases", ""),
        automation_code=result.get("automation_code", ""),
        coverage=result.get("coverage", ""),
        planner_trace=result.get("planner_trace", []),
    )
