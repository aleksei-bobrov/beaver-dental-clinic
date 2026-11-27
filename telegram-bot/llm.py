"""
Creates the chat models. This is the only file that knows which provider we use.

LangChain's init_chat_model gives every provider the same interface. To switch
from OpenRouter to, say, OpenAI or a local Ollama model: change LLM_PROVIDER and
LLM_MODEL in .env, install that provider's package (e.g. langchain-openai), and
put the API key in the variable that package reads (e.g. OPENAI_API_KEY).
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def make_model(model: str | None = None) -> BaseChatModel:
    """The main model from LLM_MODEL, or the model named in the argument."""
    provider = os.getenv("LLM_PROVIDER", "openrouter")
    model = model or os.getenv("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL is not set in .env. See .env.example.")

    extra = {}
    if provider == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY is not set in .env. See .env.example.")
        # A model on OpenRouter can be served by several providers, and not all of
        # them support tool calling. This sends requests only to providers that
        # support every parameter we use.
        extra["openrouter_provider"] = {"require_parameters": True}

    return init_chat_model(
        model,
        model_provider=provider,
        # Retry once. Free models count failed attempts against the daily limit.
        max_retries=1,
        **extra,
    )


def make_fallback_models() -> list[BaseChatModel]:
    """Backup models from LLM_FALLBACK_MODELS (comma-separated, optional).
    The agent tries them in order when the main model fails, for example when
    a free model is busy."""
    names = [name.strip() for name in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if name.strip()]
    return [make_model(name) for name in names]
