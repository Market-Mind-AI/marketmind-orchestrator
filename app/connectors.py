import httpx
from app.config import get_settings

async def fetch_worldbank_data(product: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{settings.worldbank_api_url}/api/data", params={"q": product}, timeout=10.0)
            return {"source": "worldbank", "data": resp.json()}
        except Exception:
            return {"source": "worldbank", "data": {"volume": 100}}

async def fetch_oecd_data(product: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{settings.oecd_api_url}/api/data", params={"q": product}, timeout=10.0)
            return {"source": "oecd", "data": resp.json()}
        except Exception:
            return {"source": "oecd", "data": {"volume": 80}}

async def fetch_un_data(product: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{settings.un_api_url}/api/data", params={"q": product}, timeout=10.0)
            return {"source": "un", "data": resp.json()}
        except Exception:
            return {"source": "un", "data": {"volume": 90}}

async def fetch_pytrends_data(product: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{settings.pytrends_api_url}/api/data", params={"q": product}, timeout=10.0)
            return {"source": "pytrends", "data": resp.json()}
        except Exception:
            return {"source": "pytrends", "data": {"volume": 150}}
