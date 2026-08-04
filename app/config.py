from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    worldbank_api_url: str = "http://marketmind-worldbank:8000"
    oecd_api_url: str = "http://marketmind-oecd:8000"
    un_api_url: str = "http://marketmind-united-nations-data:8000"
    pytrends_api_url: str = "http://marketmind-pytrends:8000"
    llm_api_url: str = "http://marketmind-ollma-llm:8000"
    
def get_settings():
    return Settings()
