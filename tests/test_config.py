def test_config_loads_from_env(monkeypatch):
    monkeypatch.setenv("WORLDBANK_API_URL", "http://wb")
    from app.config import get_settings
    settings = get_settings()
    assert settings.worldbank_api_url == "http://wb"
