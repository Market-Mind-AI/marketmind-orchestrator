from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Path prefix a reverse proxy serves this API under and strips before
    # forwarding — "/orchestrator" on Azure, empty locally. Without it FastAPI
    # believes it sits at the root and builds the Swagger page pointing at an
    # absolute /openapi.json, which leaves this route entirely and 404s on
    # whichever ingress rule owns "/". Routing is unaffected; only the URLs the
    # docs page generates.
    root_path: str = ""

    worldbank_api_url: str = "http://marketmind-worldbank:8000"
    oecd_api_url: str = "http://marketmind-oecd:8000"
    un_api_url: str = "http://marketmind-united-nations-data:8000"
    pytrends_api_url: str = "http://marketmind-pytrends:8000"
    llm_api_url: str = "http://marketmind-ollma-llm:8000"
    
    # Provider routing: "claude", "gpt", or "ollama"
    extraction_provider: str = "claude"
    report_provider: str = "claude"
    
    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
def get_settings():
    return Settings()
