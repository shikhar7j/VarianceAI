import pytest

from app.services.qa import DocumentQA, DocumentNotFoundError

SAMPLE_DOC = """REVENUE
Total revenue for Q3 2026 was $46.5 million, a decrease of 7% versus budget,
driven by delayed shipments in the logistics division.

MARKETING
Marketing spend was $6.1 million versus a budget of $4.0 million, driven by
an unplanned mid-quarter brand campaign.
"""


def _write_doc(tmp_path, text=SAMPLE_DOC):
    path = tmp_path / "statement.txt"
    path.write_text(text)
    return str(path)


def test_retrieves_relevant_passage(tmp_path):
    qa = DocumentQA(_write_doc(tmp_path))
    result = qa.answer("Why did revenue miss budget?")

    assert result.mode == "fallback"
    assert "revenue" in result.answer.lower()
    assert len(result.sources) >= 1


def test_unrelated_question_returns_no_match(tmp_path):
    qa = DocumentQA(_write_doc(tmp_path))
    result = qa.answer("What is the capital of France?")

    assert result.sources == []
    assert "couldn't find" in result.answer.lower()


def test_empty_question_raises(tmp_path):
    qa = DocumentQA(_write_doc(tmp_path))
    with pytest.raises(ValueError):
        qa.answer("   ")


def test_missing_document_raises(tmp_path):
    with pytest.raises(DocumentNotFoundError):
        DocumentQA(str(tmp_path / "missing.txt"))