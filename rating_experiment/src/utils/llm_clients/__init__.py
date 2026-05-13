"""Client registry for generate_ratings.py.

Two backends are supported:
  - openrouter: OpenAI-compatible REST API pointed at OpenRouter (covers all
    closed-weight frontier models and many open-weight ones via routing).
  - local: transformers + CUDA, for open-weight models you want to run on
    your own GPU rather than via a paid API.

Each backend exposes `query_llm(prompt, model_name, seed=None, temperature=1.0)`
and returns the raw generated text.
"""

from importlib import import_module
from typing import Callable

REGISTRY = {
    "openrouter": "openrouter",
    "local": "local",
}


def get_client(name: str) -> Callable[..., str]:
    if name not in REGISTRY:
        raise ValueError(f"unknown client '{name}'. options: {sorted(REGISTRY)}")
    mod = import_module(f"utils.llm_clients.{REGISTRY[name]}")
    return mod.query_llm
