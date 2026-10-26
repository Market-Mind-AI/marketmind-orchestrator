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


_TOKEN = "test-internal-token"
_HEADERS = {"X-Internal-Token": _TOKEN}


@pytest.fixture(autouse=True)
def _shared_secret() -> Generator[None, None, None]:
    """The secret the scoring service sends on /run-analysis — the same one
    this service sends back on /internal/*."""
    import app.main as main

    original = main.settings.scoring_internal_token
    main.settings.scoring_internal_token = _TOKEN
    yield
    main.settings.scoring_internal_token = original


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("CHECKPOINTER", "memory")
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


# --- starting a run costs money, so the route is not for the browser ---


def test_a_run_without_the_token_is_refused(client: TestClient) -> None:
    response = client.post("/run-analysis", json={"analysis_id": "123"})
    assert response.status_code == 401


def test_a_run_with_the_wrong_token_is_refused(client: TestClient) -> None:
    response = client.post("/run-analysis", json={"analysis_id": "123"}, headers={"X-Internal-Token": "wrong"})
    assert response.status_code == 401


def test_with_no_secret_configured_every_run_is_refused(client: TestClient) -> None:
    """Fails closed. An unset secret that waved calls through would look, from
    outside a running deployment, exactly like one configured correctly."""
    import app.main as main

    main.settings.scoring_internal_token = ""
    response = client.post("/run-analysis", json={"analysis_id": "123"}, headers=_HEADERS)
    assert response.status_code == 503


def test_a_run_without_an_analysis_id_is_refused(client: TestClient) -> None:
    """The analysis id is the run's identity. Without it, anything that retried
    would start a second run of the same analysis instead of resuming the first."""
    response = client.post("/run-analysis", json={"countries": ["ESP"]}, headers=_HEADERS)
    assert response.status_code == 422


def test_run_analysis_streams_the_failure_instead_of_closing_quietly(client: TestClient) -> None:
    response = client.post(
        "/run-analysis",
        json={"analysis_id": "123", "countries": [], "errors": [], "enrichment_results": []},
        headers=_HEADERS,
    )
    assert response.status_code == 200

    events = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert any("error" in e for e in events), "a failed run must say so on the stream"
    assert not any("mos_score" in json.dumps(e) for e in events), "no score may reach the caller"


def test_run_analysis_goes_through_the_claim(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The scoring service's resume path called this without claiming, so an
    analysis another runner was working on could execute twice."""
    from unittest.mock import AsyncMock

    from app import public

    scoring = AsyncMock()
    scoring.claim_analysis = AsyncMock(return_value=False)
    monkeypatch.setattr(public, "_client", lambda: scoring)

    response = client.post("/run-analysis", json={"analysis_id": "123"}, headers=_HEADERS)

    events = [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events == [{"skipped": "another runner holds this analysis"}]
    scoring.claim_analysis.assert_awaited_once()
