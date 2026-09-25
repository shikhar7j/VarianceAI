import csv

import pytest

from app.services.agent import FinanceAgent

STATEMENT_TEXT = """REVENUE
Total revenue for Q3 2026 was $46.5 million, a decrease of 7% versus budget.

RISK FACTORS
Key risks disclosed for the upcoming quarter include supply chain volatility.
"""


@pytest.fixture
def agent(tmp_path):
    csv_path = tmp_path / "budget.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["line_item", "department", "budget", "actual"])
        writer.writeheader()
        writer.writerow({"line_item": "Marketing Spend", "department": "Marketing",
                          "budget": 100, "actual": 160})

    doc_path = tmp_path / "statement.txt"
    doc_path.write_text(STATEMENT_TEXT)

    return FinanceAgent(str(csv_path), str(doc_path))


def test_budget_keyword_routes_to_variance_tool(agent):
    result = agent.ask("What line items went over budget?")
    assert result["route"] == "variance"
    assert result["tool_used"] == "variance_tool"
    assert "Marketing" in result["answer"]


def test_narrative_question_routes_to_qa_tool(agent):
    result = agent.ask("What risks are disclosed for next quarter?")
    assert result["route"] == "document_qa"
    assert result["tool_used"] == "qa_tool"


def test_empty_question_raises(agent):
    with pytest.raises(ValueError):
        agent.ask("   ")


def test_fallback_mode_reported(agent):
    result = agent.ask("What was the variance on marketing?")
    assert result["mode"] == "fallback"