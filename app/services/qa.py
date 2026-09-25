from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app import llm_client
from app.config import settings

logger = logging.getLogger(__name__)

CHUNK_SIZE_CHARS = 500
CHUNK_OVERLAP_CHARS = 50
MIN_SIMILARITY = 0.05
TOP_K = 3


class DocumentNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class QAResult:
    answer: str
    sources: list[str]
    mode: str  # "live" | "fallback"

    def to_dict(self) -> dict:
        return {"answer": self.answer, "sources": self.sources, "mode": self.mode}


def _chunk_text(text: str) -> list[str]:
    """Split into overlapping chunks using LangChain's recursive splitter.

    Recursive splitting tries paragraph breaks first, then sentences, then
    words, so chunks stay semantically coherent instead of cutting mid-thought
    the way a naive fixed-word split would.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


class DocumentQA:
    def __init__(self, doc_path: str):
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except FileNotFoundError as exc:
            raise DocumentNotFoundError(f"Document not found: {doc_path}") from exc

        self._chunks = _chunk_text(raw_text)
        if not self._chunks:
            raise ValueError(f"Document at {doc_path} produced no usable text chunks")

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._chunk_vectors = self._vectorizer.fit_transform(self._chunks)

    def _retrieve(self, question: str, k: int = TOP_K) -> list[str]:
        query_vector = self._vectorizer.transform([question])
        similarities = cosine_similarity(query_vector, self._chunk_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:k]
        return [self._chunks[i] for i in top_indices if similarities[i] > MIN_SIMILARITY]

    def _synthesize(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(context_chunks)
        prompt = (
            "Answer the question using ONLY the context below, from a company's "
            "financial statement. If the context doesn't fully answer it, say so.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        )
        try:
            return llm_client.complete(prompt, max_tokens=200, temperature=0.2)
        except Exception:
            logger.warning("LLM synthesis failed, falling back to raw passage")
            return context_chunks[0]

    def answer(self, question: str) -> QAResult:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        mode = "live" if settings.live_mode else "fallback"
        retrieved = self._retrieve(question)

        if not retrieved:
            return QAResult(
                answer="I couldn't find anything relevant to that question in the document.",
                sources=[],
                mode=mode,
            )

        answer_text = self._synthesize(question, retrieved) if settings.live_mode else retrieved[0]
        return QAResult(answer=answer_text, sources=retrieved, mode=mode)