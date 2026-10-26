def test_config_loads_from_env(monkeypatch):
    monkeypatch.setenv("SCORING_API_URL", "http://scoring")
    from app.config import get_settings

    settings = get_settings()
    assert settings.scoring_api_url == "http://scoring"


def test_config_ignores_settings_it_no_longer_has(monkeypatch):
    # Deployments may still carry the collectors' addresses from before every
    # call went through the scoring service. They must not break startup.
    monkeypatch.setenv("WORLDBANK_API_URL", "http://wb")
    from app.config import get_settings

    settings = get_settings()
    assert not hasattr(settings, "worldbank_api_url")
