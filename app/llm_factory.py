from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from app.config import get_settings

def get_llm(provider: str) -> BaseChatModel:
    settings = get_settings()
    code = provider.strip().lower()
    
    if code == "claude":
        if not settings.anthropic_api_key:
            raise ValueError("anthropic_api_key is not set in config")
        return ChatAnthropic(
            model_name="claude-3-5-sonnet-20240620", 
            api_key=settings.anthropic_api_key
        )
    elif code == "gpt":
        if not settings.openai_api_key:
            raise ValueError("openai_api_key is not set in config")
        return ChatOpenAI(
            model="gpt-4o", 
            api_key=settings.openai_api_key
        )
    elif code == "ollama":
        # A simple fallback for Ollama using ChatOpenAI wrapper pointing to local server
        return ChatOpenAI(
            model="qwen3-vl:8b-instruct",
            api_key="ollama", # required but ignored
            base_url=f"{settings.llm_api_url}/v1"
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
