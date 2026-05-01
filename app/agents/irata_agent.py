from __future__ import annotations

from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from app.core.rag import RAGStore
from app.core.tools import coverage_check


class IRATAState(TypedDict, total=False):
    question: str
    requirement_summary: str
    retrieved_context: str
    testcases: str
    coverage: str
    automation_code: str
    framework: str
    planner_trace: list[str]


def build_irata_graph(llm, store: RAGStore):
    def plan(state: IRATAState) -> IRATAState:
        question = state.get("question", "")
        if not llm:
            trace = [
                f"Plan: retrieve requirements for '{question}'",
                "Plan: summarize requirements and dependencies",
                "Plan: generate functional/negative/boundary/edge tests",
                "Plan: check coverage and iterate if low",
                "Plan: generate automation code",
            ]
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Draft a short tool plan for requirement analysis and test automation."),
                    ("user", "Question: {question}"),
                ]
            )
            response = llm.invoke(prompt.format_messages(question=question)).content
            trace = [line.strip("- ") for line in response.splitlines() if line.strip()]
        return {"planner_trace": trace}
    def retrieve(state: IRATAState) -> IRATAState:
        docs = store.query(state["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"retrieved_context": context}

    def understand(state: IRATAState) -> IRATAState:
        if not llm:
            summary = "Summary unavailable (LLM not configured)."
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Summarize the requirement and dependencies."),
                    ("user", "{context}"),
                ]
            )
            summary = llm.invoke(
                prompt.format_messages(context=state.get("retrieved_context", ""))
            ).content
        return {"requirement_summary": summary}

    def generate_tests(state: IRATAState) -> IRATAState:
        if not llm:
            tests = "| ID | Scenario | Steps | Expected |\n| --- | --- | --- | --- |\n| TC01 | Sample | Step | Expected |"
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Create functional, negative, boundary, and edge tests in a markdown table."),
                    ("user", "{summary}"),
                ]
            )
            tests = llm.invoke(
                prompt.format_messages(summary=state.get("requirement_summary", ""))
            ).content
        return {"testcases": tests}

    def evaluate_coverage(state: IRATAState) -> IRATAState:
        return {"coverage": coverage_check(state.get("testcases", ""))}

    def generate_automation(state: IRATAState) -> IRATAState:
        if not llm:
            code = "# Automation code unavailable (LLM not configured)."
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Generate minimal automation code using the framework."),
                    ("user", "Framework: {framework}\nTests:\n{tests}"),
                ]
            )
            code = llm.invoke(
                prompt.format_messages(
                    framework=state.get("framework", "Playwright + pytest"),
                    tests=state.get("testcases", ""),
                )
            ).content
        return {"automation_code": code}

    def decide_next(state: IRATAState) -> str:
        if state.get("coverage") == "low":
            return "generate_tests"
        return "generate_automation"

    graph = StateGraph(IRATAState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("understand", understand)
    graph.add_node("generate_tests", generate_tests)
    graph.add_node("evaluate_coverage", evaluate_coverage)
    graph.add_node("generate_automation", generate_automation)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "understand")
    graph.add_edge("understand", "generate_tests")
    graph.add_edge("generate_tests", "evaluate_coverage")
    graph.add_conditional_edges("evaluate_coverage", decide_next)
    graph.add_edge("generate_automation", END)

    return graph.compile()
