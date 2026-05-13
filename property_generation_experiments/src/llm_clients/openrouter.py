"""OpenRouter client using the OpenAI Python SDK.

Set OPENROUTER_API_KEY in your environment (or .env at the repo root).
Pass any OpenRouter-supported model id, e.g.:
  - openai/gpt-5.4
  - anthropic/claude-sonnet-4-6
  - google/gemini-2.5-flash-lite
  - meta-llama/llama-3.1-70b-instruct
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_BASE_URL = "https://openrouter.ai/api/v1"
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment or .env")
        _client = OpenAI(api_key=api_key, base_url=_BASE_URL)
    return _client


def query_llm(prompt: str, model_name: str, seed: int = None,
              temperature: float = 1.0, max_tokens: int = 512) -> str:
    client = _get_client()
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        kwargs["seed"] = seed

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
