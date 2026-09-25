
from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from app.config import settings
from app.services.qa import DocumentQA, DocumentNotFoundError
from app.services.variance import VarianceAnalyzer, VarianceDataError

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)

_qa_engine: DocumentQA | None = None


def _get_qa_engine() -> DocumentQA:
    global _qa_engine
    if _qa_engine is None:
        _qa_engine = DocumentQA(settings.financial_statement_path)
    return _qa_engine


@bp.route("/")
def index():
    return send_from_directory(current_app.static_folder, "index.html")


@bp.route("/api/status")
def status():
    return jsonify(
        {
            "mode": "live" if settings.live_mode else "fallback",
            "model": settings.openai_model if settings.live_mode else None,
        }
    )


@bp.route("/api/variance")
def variance():
    try:
        analyzer = VarianceAnalyzer(settings.budget_actuals_path)
        results = analyzer.run()
        return jsonify([r.to_dict() for r in results])
    except VarianceDataError as exc:
        logger.error("Variance data error: %s", exc)
        return jsonify({"error": str(exc)}), 422
    except Exception:
        logger.exception("Unexpected error computing variance")
        return jsonify({"error": "internal server error"}), 500


@bp.route("/api/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "question (non-empty string) is required"}), 400

    try:
        engine = _get_qa_engine()
        result = engine.answer(question)
        return jsonify(result.to_dict())
    except DocumentNotFoundError as exc:
        logger.error("Document not found: %s", exc)
        return jsonify({"error": "reference document not found on server"}), 500
    except Exception:
        logger.exception("Unexpected error answering question")
        return jsonify({"error": "internal server error"}), 500