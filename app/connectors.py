import httpx
from app.config import get_settings

async def fetch_worldbank_data(product: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{settings.worldbank_api_url}/api/data", params={"q": product})
        return {"source": "worldbank", "data": resp.json()}

async def fetch_oecd_data(product: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{settings.oecd_api_url}/api/data", params={"q": product})
        return {"source": "oecd", "data": resp.json()}

async def fetch_un_data(product: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{settings.un_api_url}/api/data", params={"q": product})
        return {"source": "un", "data": resp.json()}

async def fetch_pytrends_data(product: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{settings.pytrends_api_url}/api/data", params={"q": product})
        return {"source": "pytrends", "data": resp.json()}
