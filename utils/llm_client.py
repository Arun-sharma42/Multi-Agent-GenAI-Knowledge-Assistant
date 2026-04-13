from langchain_core.language_models import BaseLanguageModel
from utils.config import config
from utils.logger import get_logger

log = get_logger("LLMClient")


def get_llm(temperature: float = 0.2) -> BaseLanguageModel:
    provider = config.LLM_PROVIDER.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        log.info(f"Initialising Anthropic Claude → {config.LLM_MODEL}")

        return ChatAnthropic(
            model=config.LLM_MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        #  FORCE OLD SUPPORTED MODEL
        model_name = "gemini-1.5-flash"

        log.info(f"Initialising Google Gemini -> {model_name}")

        return ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Set LLM_PROVIDER=anthropic or LLM_PROVIDER=google in your env."
        )