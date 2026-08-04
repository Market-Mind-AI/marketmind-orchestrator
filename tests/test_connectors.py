import pytest
import httpx
from app.connectors import fetch_worldbank_data

@pytest.mark.asyncio
async def test_fetch_worldbank(respx_mock):
    respx_mock.get("http://marketmind-worldbank:8000/api/data?q=test").mock(return_value=httpx.Response(200, json={"data": "wb"}))
    res = await fetch_worldbank_data("test")
    assert res == {"source": "worldbank", "data": {"data": "wb"}}
