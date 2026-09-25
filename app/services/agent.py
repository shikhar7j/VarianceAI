
from __future__ import annotations

import logging
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, END

from app import llm_client
from app.config import settings
from app.services.qa import DocumentQA
from app.services.variance import VarianceAnalyzer

logger = logging.getLogger(__name__)

Route = Literal["variance", "document_qa"]

VARIANCE_KEYWORDS = {
    "budget", "variance", "actual", "overspend", "underspend",
    "over budget", "under budget", "spend", "cost", "expense",
}


class AgentState(TypedDict, total=False):
    question: str
    route: Route
    answer: str
    tool_used: str


class FinanceAgent:
    """Two-tool LangGraph agent: variance analysis and document Q&A."""

    def __init__(self, budget_csv_path: str, statement_path: str):
        self._variance_analyzer = VarianceAnalyzer(budget_csv_path)
        self._qa_engine = DocumentQA(statement_path)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self._route_node)
        graph.add_node("variance_tool", self._variance_node)
        graph.add_node("qa_tool", self._qa_node)

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {"variance": "variance_tool", "document_qa": "qa_tool"},
        )
        graph.add_edge("variance_tool", END)
        graph.add_edge("qa_tool", END)

        return graph.compile()

    def _route_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        route = self._classify_llm(question) if settings.live_mode else self._classify_heuristic(question)
        logger.info("Routed question to %s: %r", route, question)
        return {**state, "route": route}

    def _variance_node(self, state: AgentState) -> AgentState:
        results = self._variance_analyzer.run()
        flagged = [r for r in results if abs(r.variance_pct) >= 10]
        summary = " ".join(r.commentary for r in flagged) if flagged else "No material variances this period."
        return {**state, "answer": summary, "tool_used": "variance_tool"}

    def _qa_node(self, state: AgentState) -> AgentState:
        result = self._qa_engine.answer(state["question"])
        return {**state, "answer": result.answer, "tool_used": "qa_tool"}


    @staticmethod
    def _classify_heuristic(question: str) -> Route:
        q = question.lower()
        return "variance" if any(kw in q for kw in VARIANCE_KEYWORDS) else "document_qa"

    @staticmethod
    def _classify_llm(question: str) -> Route:
        prompt = (
            "Classify the question below into exactly one category. "
            "Reply with only the category name, nothing else.\n\n"
            "- 'variance': the question is about budget vs actual numbers, "
            "overspend, underspend, or line-item variances.\n"
            "- 'document_qa': the question asks about narrative context, "
            "explanations, or risks described in a financial statement.\n\n"
            f"Question: {question}\nCategory:"
        )
        try:
            raw = llm_client.complete(prompt, max_tokens=10, temperature=0.0)
            cleaned = raw.strip().lower()
            return "variance" if "variance" in cleaned else "document_qa"
        except Exception:
            logger.warning("LLM routing failed, falling back to heuristic")
            return FinanceAgent._classify_heuristic(question)


    def ask(self, question: str) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        result = self._graph.invoke({"question": question})
        return {
            "answer": result["answer"],
            "route": result["route"],
            "tool_used": result["tool_used"],
            "mode": "live" if settings.live_mode else "fallback",
        }