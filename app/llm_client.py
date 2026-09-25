from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def get_client():
    """Return a cached OpenAI client, or None if no API key is configured."""
    global _client
    if not settings.live_mode:
        return None
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def complete(prompt: str, *, max_tokens: int = 200, temperature: float = 0.3) -> str:
    """Send a single-turn completion request. Caller must check live_mode first."""
    client = get_client()
    if client is None:
        raise RuntimeError("complete() called without a configured OpenAI client")

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception("OpenAI completion request failed")
        raise