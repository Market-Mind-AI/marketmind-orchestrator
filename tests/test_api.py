import json
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture(autouse=True)
def _in_memory_checkpointer() -> Generator[None, None, None]:
    """Asked for by name, because losing runs on restart is only ever
    acceptable in a test."""
    get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None
    import app.main as main

    original = main.settings.checkpointer
    main.settings.checkpointer = "memory"
    yield
    main.settings.checkpointer = original


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("CHECKPOINTER", "memory")
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_a_run_without_an_analysis_id_is_refused(client: TestClient) -> None:
    """The analysis id is the run's identity. Without it, anything that retried
    would start a second run of the same analysis instead of resuming the first."""
    response = client.post("/run-analysis", json={"countries": ["ESP"]})
    assert response.status_code == 422


def test_run_analysis_streams_the_failure_instead_of_closing_quietly(client: TestClient) -> None:
    response = client.post(
        "/run-analysis",
        json={"analysis_id": "123", "countries": [], "errors": [], "enrichment_results": []},
    )
    assert response.status_code == 200

    events = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert any("error" in e for e in events), "a failed run must say so on the stream"
    assert not any("mos_score" in json.dumps(e) for e in events), "no score may reach the caller"
