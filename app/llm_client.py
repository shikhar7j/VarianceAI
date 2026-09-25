from __future__ import annotations
 
import logging
 
from app.config import settings
 
logger = logging.getLogger(__name__)
 
_llm = None
 
 
def get_llm():
    """Return a cached LangChain ChatOpenAI instance, or None if no key is set."""
    global _llm
    if not settings.live_mode:
        return None
    if _llm is None:
        from langchain_openai import ChatOpenAI
 
        _llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
        )
    return _llm
 
 
def complete(prompt: str, *, max_tokens: int = 200, temperature: float = 0.3) -> str:
    """Send a single-turn completion request. Caller must check live_mode first."""
    llm = get_llm()
    if llm is None:
        raise RuntimeError("complete() called without a configured LLM client")
 
    try:
        response = llm.invoke(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content.strip()
    except Exception:
        logger.exception("LLM completion request failed")
        raise
 