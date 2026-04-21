"""
utils/llm_client.py
--------------------
LLM factory (clean & stable version)
"""

from langchain_core.language_models import BaseLanguageModel
from utils.config import config
from utils.logger import get_logger

log = get_logger("LLMClient")


def get_llm(temperature: float = 0.2) -> BaseLanguageModel:
    provider = config.LLM_PROVIDER.lower()

    # -- GOOGLE GEMINI -------------------------------------
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        log.info(f"Using Gemini Model -> {config.LLM_MODEL}")

        return ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,              # ✅ from config only
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True,  # Required for Gemini
        )

    # -- ANTHROPIC CLAUDE ----------------------------------
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        log.info(f"Using Claude -> {config.LLM_MODEL}")

        return ChatAnthropic(
            model=config.LLM_MODEL,
            anthropic_api_key=getattr(config, "ANTHROPIC_API_KEY", ""),
            temperature=temperature,
        )

    # -- ERROR ---------------------------------------------
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}"
        )