"""The endpoints the browser talks to, now that the agent is the way in."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app import public
from app.main import app
from app.scoring_client import ScoringRejected, ScoringUnavailable


@pytest.fixture
def scoring(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    client = AsyncMock()
    client.resolve_product = AsyncMock(
        return_value={"entity_key": "e1", "canonical_title": "air fryer"}
    )
    client.register_analysis = AsyncMock(return_value={"request_id": "r1", "status": "pending"})
    client.read_report = AsyncMock(return_value={"request_id": "r1", "status": "done"})
    client.countries = AsyncMock(return_value=[{"iso3": "ESP", "name": "Spain"}])
    client.identify_image = AsyncMock(return_value={"product_name": "air fryer"})
    monkeypatch.setattr(public, "_client", lambda: client)
    return client


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The run is a background task, so the test records that it was started
    rather than waiting for a graph it did not build."""
    ids: list[str] = []

    async def fake_run(analysis_id: str) -> None:
        ids.append(analysis_id)

    monkeypatch.setattr(public, "_run", fake_run)
    return ids



def _client() -> TestClient:
    return TestClient(app)


def test_analyze_resolves_then_opens_the_record_then_starts_the_run(
    scoring: AsyncMock, started: list[str]
) -> None:
    response = _client().post(
        "/analyze",
        json={"product_name": "air fryer", "countries": ["ESP", "PRT"], "brand": "Philips"},
    )

    assert response.status_code == 202
    assert response.json() == {"request_id": "r1", "status": "pending"}
    # 202 and not 200: the run takes minutes, and a request held open for it is
    # a request a proxy will cut.
    scoring.resolve_product.assert_awaited_once()
    assert scoring.resolve_product.await_args.kwargs["brand"] == "Philips"
    scoring.register_analysis.assert_awaited_once()
    assert started == ["r1"]


def test_the_curated_vocabulary_is_carried_into_the_resolve(
    scoring: AsyncMock, started: list[str]
) -> None:
    """The identify step already produced it, with whatever the buyer corrected.
    Dropping it here would throw that work away and make the enrichment pay for
    it a second time."""
    markets = {"ESP": {"primary_name": "freidora de aire", "search_terms": ["freidora de aire"]}}
    candidates = [{"country": "USA", "primary_name": "air fryer"}]

    _client().post(
        "/analyze",
        json={
            "product_name": "air fryer",
            "countries": ["ESP"],
            "localized_markets": markets,
            "reference_candidates": candidates,
        },
    )

    kwargs = scoring.resolve_product.await_args.kwargs
    assert kwargs["localized_markets"] == markets
    assert kwargs["reference_candidates"] == candidates


def test_no_product_and_no_markets_are_refused_before_anything_is_opened(
    scoring: AsyncMock, started: list[str]
) -> None:
    """A record opened for a request that cannot be answered is a record that
    gets billed for nothing."""
    client = _client()

    assert client.post("/analyze", json={"countries": ["ESP"]}).status_code == 422
    assert client.post("/analyze", json={"product_name": "air fryer"}).status_code == 422

    scoring.register_analysis.assert_not_awaited()
    assert started == []


def test_a_scoring_outage_answers_503_and_a_rejection_answers_its_own_status(
    scoring: AsyncMock, started: list[str]
) -> None:
    """The browser is owed the difference: one is worth retrying, the other is
    the request being wrong."""
    scoring.resolve_product = AsyncMock(side_effect=ScoringUnavailable("down"))
    assert _client().post("/analyze", json={"product_name": "x", "countries": ["ESP"]}).status_code == 503

    scoring.resolve_product = AsyncMock(side_effect=ScoringRejected(422, "unknown market"))
    response = _client().post("/analyze", json={"product_name": "x", "countries": ["ZZZ"]})
    assert response.status_code == 422
    assert response.json()["detail"] == "unknown market"


def test_nothing_is_opened_when_the_product_cannot_be_resolved(
    scoring: AsyncMock, started: list[str]
) -> None:
    """The failure people saw: the product came back named and the analysis
    failed anyway, because resolving happened somewhere else from running. Both
    are here now, so a resolve that fails opens no record and starts no run."""
    scoring.resolve_product = AsyncMock(side_effect=ScoringUnavailable("down"))

    _client().post("/analyze", json={"product_name": "air fryer", "countries": ["ESP"]})

    scoring.register_analysis.assert_not_awaited()
    assert started == []


def test_reading_an_analysis_is_never_served_from_a_cache(scoring: AsyncMock) -> None:
    response = _client().get("/analyze/9f1d2c3b-0000-4000-8000-000000000000")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_an_id_that_is_not_a_uuid_is_refused(scoring: AsyncMock) -> None:
    assert _client().get("/analyze/not-a-uuid").status_code == 422
    assert _client().get("/analyze/not-a-uuid/stream").status_code == 422
    scoring.read_report.assert_not_awaited()


def test_countries_and_identify_are_passed_through(scoring: AsyncMock) -> None:
    client = _client()

    assert client.get("/countries").json() == [{"iso3": "ESP", "name": "Spain"}]

    response = client.post("/catalog/identify-image", json={"image_base64": "ZmFrZQ=="})
    assert response.status_code == 200
    scoring.identify_image.assert_awaited_once()


def test_a_run_whose_graph_blew_up_is_recorded_as_failed(scoring: AsyncMock) -> None:
    """There is no caller left to raise to, so an unhandled error has to land on
    the record — otherwise the analysis stays `running` forever and the buyer
    waits for something that already stopped."""
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("app.graph.get_graph", return_value=graph):
        import asyncio

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(public._run("r1"))

    scoring.fail_analysis.assert_awaited_once()


async def test_a_run_in_flight_is_held_onto(scoring: AsyncMock) -> None:
    """asyncio keeps only a weak reference to a task, so one nobody holds can be
    collected mid-run and cancelled. The analysis would just stop, with the
    record still saying `running` — rare, silent, and impossible to reproduce on
    demand."""
    import asyncio

    release = asyncio.Event()

    async def slow(analysis_id: str) -> None:
        await release.wait()

    with patch.object(public, "_run", slow):
        public._start("r1")
        assert len(public._in_flight) == 1  # held while it runs
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert public._in_flight == set()  # and let go when it finishes
