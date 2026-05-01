# Intelligent Requirement Analysis & Test Automation Agent (IRATA)

Transforms requirement documents into test intelligence automatically.

## What this demo includes

- FastAPI service with ingestion and query endpoints
- Vector store RAG (FAISS + sentence-transformer embeddings)
- LangGraph orchestration with coverage loop
- Minimal LangChain + Groq LLM integration
- Sample requirements for quick demo

## Architecture (interview-friendly)

- Frontend: dashboard/chatbot (not included)
- Backend: FastAPI
- Pipeline: LangChain RAG core
- Agent: LangGraph
- Storage: FAISS in-memory index (swap for Chroma/PGVector later)
- LLM: Groq (configurable)
- Monitoring: LangSmith (optional)

## Quickstart

1) Create a virtual environment and install dependencies

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure environment variables

```
cp .env.example .env
# Add GROQ_API_KEY when available
```

3) Run the API

```
uvicorn app.main:app --reload
```

## Demo flow

1) Ingest sample requirements

```
curl -X POST http://localhost:8000/ingest \
	-H "Content-Type: application/json" \
	-d '{"source":"sample_requirements","text":"Login requirements: password min 8 chars."}'
```

2) Query and generate tests

```
curl -X POST http://localhost:8000/query \
	-H "Content-Type: application/json" \
	-d '{"question":"What are login requirements?","framework":"Playwright + pytest"}'
```

3) Upload a PDF/DOCX/CSV/XLSX/TXT

```
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@path/to/requirements.pdf"
```

## Notes

- If `GROQ_API_KEY` is not set, the agent returns placeholder summaries/tests.
- FAISS runs in-memory for the demo; add persistence for production.
- Supported file types: PDF, DOCX, TXT, CSV, XLSX.