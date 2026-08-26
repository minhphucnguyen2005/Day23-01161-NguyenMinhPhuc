"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()


def get_llm(model: str | None = None, temperature: float = 0.0) -> Any:  # noqa: ANN401
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY -> ChatGoogleGenerativeAI
    2. OPENAI_API_KEY -> ChatOpenAI
    3. ANTHROPIC_API_KEY -> ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and not gemini_key.startswith("AIza..."):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        model_name = str(model or os.getenv("LLM_MODEL") or "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_key,
            temperature=temperature,
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith("sk-..."):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        model_name = str(model or os.getenv("LLM_MODEL") or "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and not anthropic_key.startswith("sk-ant-..."):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        model_name = str(model or os.getenv("LLM_MODEL") or "claude-3-5-sonnet-latest")
        return ChatAnthropic(  # type: ignore[call-arg]
            model_name=model_name,
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
