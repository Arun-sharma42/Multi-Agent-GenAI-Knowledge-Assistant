"""
utils/llm_client.py
────────────────────
Thin wrapper that returns a LangChain-compatible LLM object.
Supports both Anthropic Claude and OpenAI — switch via .env.

Interview talking point:
  "I abstracted the LLM behind a factory function so the rest of the
   system is provider-agnostic. Swapping Claude for GPT-4o is one
   env-var change."
"""

from langchain_core.language_models import BaseLanguageModel
from utils.config import config
from utils.logger import get_logger

log = get_logger("LLMClient")


def get_llm(temperature: float = 0.2) -> BaseLanguageModel:
    """
    Factory function — returns a ready-to-use LangChain LLM.

    Args:
        temperature: 0 = deterministic (good for SQL), 0.7 = creative.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        log.info(f"Initialising Anthropic Claude → {config.LLM_MODEL}")
        return ChatAnthropic(
            model=config.LLM_MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        log.info(f"Initialising OpenAI → {config.LLM_MODEL}")
        return ChatOpenAI(
            model=config.LLM_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Set LLM_PROVIDER=anthropic or LLM_PROVIDER=openai in .env"
        )
