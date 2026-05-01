from __future__ import annotations

from pydantic import BaseModel


class IngestRequest(BaseModel):
    source: str
    text: str


class QueryRequest(BaseModel):
    question: str
    framework: str = "Playwright + pytest"


class QueryResponse(BaseModel):
    requirement_summary: str
    testcases: str
    automation_code: str
    coverage: str
    planner_trace: list[str]
