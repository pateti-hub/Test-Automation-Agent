from __future__ import annotations

from langchain_core.tools import tool

from app.core.rag import RAGStore


@tool
def retrieve_requirement(question: str, store: RAGStore) -> str:
    """Retrieve requirement context relevant to a question."""
    docs = store.query(question)
    if not docs:
        return "No matching requirement found."
    sources = {doc.metadata.get("source", "unknown") for doc in docs}
    content = "\n\n".join(doc.page_content for doc in docs)
    return f"Sources: {', '.join(sorted(sources))}\n{content}"


@tool
def generate_testcases(requirement_summary: str) -> str:
    """Generate a minimal set of test cases from a requirement summary."""
    return (
        "| ID | Scenario | Steps | Expected |\n"
        "| --- | --- | --- | --- |\n"
        "| TC01 | Valid input | Provide valid values | Success |\n"
        "| TC02 | Empty input | Leave required fields blank | Error displayed |\n"
        f"| TC03 | Boundary | Use min/max values from: {requirement_summary} | Pass/Fail |"
    )


@tool
def generate_automation_code(testcases_markdown: str, framework: str) -> str:
    """Generate automation code based on test cases and framework."""
    if "playwright" in framework.lower():
        return (
            "import pytest\n"
            "from playwright.async_api import Page\n\n"
            "@pytest.mark.asyncio\n"
            "async def test_valid_flow(page: Page):\n"
            "    await page.goto(\"/login\")\n"
            "    await page.fill(\"#username\", \"demo\")\n"
            "    await page.fill(\"#password\", \"Passw0rd!\")\n"
            "    await page.click(\"#submit\")\n"
            "    await page.wait_for_url(\"**/dashboard\")\n"
        )
    return (
        "import pytest\n\n"
        "def test_valid_flow():\n"
        "    assert True\n"
    )


@tool
def fetch_jira_ticket(ticket_id: str) -> str:
    """Stub Jira ticket fetch for demo purposes."""
    return f"Stub Jira ticket payload for {ticket_id}."


@tool
def analyze_bug_log(log_text: str) -> str:
    """Analyze bug logs and return a brief clustering summary."""
    return "Cluster: validation errors. Root cause: missing null checks."


@tool
def coverage_check(testcases_markdown: str) -> str:
    """Return a simple coverage status based on test cases content."""
    if "Boundary" not in testcases_markdown:
        return "low"
    return "high"
