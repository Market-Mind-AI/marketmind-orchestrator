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
    client.register_analysis = AsyncMock(
        return_value={"request_id": "r1", "status": "pending", "entity_key": "e1", "canonical_title": "air fryer"}
    )
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


def test_analyze_registers_in_one_call_then_starts_the_run(
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
    scoring.register_analysis.assert_awaited_once()
    assert scoring.register_analysis.await_args.args[0]["brand"] == "Philips"
    assert started == ["r1"]


def test_the_curated_vocabulary_is_carried_into_the_registration(
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

    kwargs = scoring.register_analysis.await_args.args[0]
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
    scoring.register_analysis = AsyncMock(side_effect=ScoringUnavailable("down"))
    assert _client().post("/analyze", json={"product_name": "x", "countries": ["ESP"]}).status_code == 503

    scoring.register_analysis = AsyncMock(side_effect=ScoringRejected(422, "unknown market"))
    response = _client().post("/analyze", json={"product_name": "x", "countries": ["ZZZ"]})
    assert response.status_code == 422
    assert response.json()["detail"] == "unknown market"


def test_no_run_starts_when_the_product_cannot_be_registered(
    scoring: AsyncMock, started: list[str]
) -> None:
    """The failure people saw: the product came back named and the analysis
    failed anyway, because resolving happened somewhere else from running. A
    registration that fails starts no run."""
    scoring.register_analysis = AsyncMock(side_effect=ScoringUnavailable("down"))

    _client().post("/analyze", json={"product_name": "air fryer", "countries": ["ESP"]})

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


def _graph(*, events=(), error: Exception | None = None, delivered: bool = False):
    """A compiled graph's surface as _run uses it: astream, then the checkpoint."""
    from types import SimpleNamespace

    graph = AsyncMock()

    async def astream(_payload, config=None):
        for event in events:
            yield event
        if error is not None:
            raise error

    graph.astream = astream
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={"delivered": delivered}))
    return graph


async def test_a_run_whose_graph_blew_up_is_recorded_as_failed(scoring: AsyncMock) -> None:
    """There is no caller left to raise to, so an unhandled error has to land on
    the record — otherwise the analysis stays `running` forever and the buyer
    waits for something that already stopped."""
    with patch("app.graph.get_graph", return_value=_graph(error=RuntimeError("boom"))):
        await public._run("r1")

    scoring.fail_analysis.assert_awaited_once()


async def test_a_delivered_analysis_is_never_failed_afterwards(scoring: AsyncMock) -> None:
    """The buyer has the answer and was billed for it. An error after delivery —
    the narrative, a measurement — broke a reading of it, not the answer."""
    graph = _graph(events=[{"deliver": {"delivered": True}}], error=RuntimeError("boom"), delivered=True)

    with patch("app.graph.get_graph", return_value=graph):
        await public._run("r1")

    scoring.fail_analysis.assert_not_awaited()


async def test_a_failing_cost_report_does_not_fail_the_analysis(scoring: AsyncMock) -> None:
    with (
        patch("app.graph.get_graph", return_value=_graph(events=[{"deliver": {"delivered": True}}])),
        patch("app.metrics.analysis_cost", side_effect=RuntimeError("pricing table")),
    ):
        await public._run("r1")

    scoring.fail_analysis.assert_not_awaited()


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


async def test_orphans_left_by_a_restart_are_resumed_immediately(scoring: AsyncMock) -> None:
    """The whole point. This process knows it just started; what it cannot know
    is which analyses its predecessor was running — so it asks. Before this, an
    orphan waited fifteen minutes for the scoring service's sweep, with the
    buyer watching a screen that says "in progress" and nothing progressing."""
    scoring.orphaned_analyses = AsyncMock(return_value=["a1", "a2"])
    scoring.claim_analysis = AsyncMock(return_value=True)
    started: list[str] = []

    with patch.object(public, "_start", started.append):
        resumed = await public.resume_orphans()

    assert resumed == 2
    assert started == ["a1", "a2"]


async def test_an_analysis_another_agent_is_running_is_left_alone(scoring: AsyncMock) -> None:
    """This is what makes a second replica safe. Without the claim, a pod coming
    up would take over what a live pod is working on and run it twice."""
    scoring.orphaned_analyses = AsyncMock(return_value=["mine", "theirs"])
    scoring.claim_analysis = AsyncMock(side_effect=[True, False])
    started: list[str] = []

    with patch.object(public, "_start", started.append):
        resumed = await public.resume_orphans()

    assert resumed == 1
    assert started == ["mine"]


async def test_a_scoring_service_that_is_not_up_yet_resumes_nothing_and_does_not_crash(
    scoring: AsyncMock,
) -> None:
    """Startup order is not guaranteed. A resume that raises here takes the
    whole service down, and the sweep is still there as the net underneath."""
    scoring.orphaned_analyses = AsyncMock(side_effect=ScoringUnavailable("not up"))

    assert await public.resume_orphans() == 0


async def test_a_run_is_claimed_before_it_runs(scoring: AsyncMock) -> None:
    """Otherwise a pod restarting elsewhere sees it unclaimed and takes it over
    while this process is working on it. Every start goes through here —
    /analyze, the orphan watch and /run-analysis alike."""
    graph = _graph()

    with patch("app.graph.get_graph", return_value=graph):
        await public._run("r1")

    scoring.claim_analysis.assert_awaited()
    assert scoring.claim_analysis.await_args_list[0].kwargs["runner"] == public.runner_id()


async def test_a_run_held_by_another_runner_does_not_execute(scoring: AsyncMock) -> None:
    """The resume path that bypassed the claim could run one analysis twice."""
    scoring.claim_analysis = AsyncMock(return_value=False)
    heard: list = []

    with patch("app.graph.get_graph", side_effect=AssertionError("the graph must not run")):
        await public._run("r1", heard.append)

    assert heard == [{"skipped": "another runner holds this analysis"}, None]
    scoring.fail_analysis.assert_not_awaited()


async def test_a_claim_that_cannot_be_asked_about_does_not_run(scoring: AsyncMock) -> None:
    """Running without knowing is how one analysis ends up running twice. The
    next orphan pass asks again."""
    scoring.claim_analysis = AsyncMock(side_effect=ScoringUnavailable("down"))
    heard: list = []

    with patch("app.graph.get_graph", side_effect=AssertionError("the graph must not run")):
        await public._run("r1", heard.append)

    assert "error" in heard[0]
    assert heard[-1] is None


async def test_the_same_analysis_never_runs_twice_in_one_process(scoring: AsyncMock) -> None:
    """A runner re-claiming its own analysis is granted — that is how it renews
    — so the claim alone cannot stop a second start inside this process."""
    import asyncio

    release = asyncio.Event()

    async def astream(_payload, config=None):
        await release.wait()
        yield {"score": {}}

    graph = _graph()
    graph.astream = astream
    heard: list = []

    with patch("app.graph.get_graph", return_value=graph):
        first = asyncio.create_task(public._run("r1"))
        await asyncio.sleep(0.01)
        await public._run("r1", heard.append)
        release.set()
        await first

    assert heard == [{"skipped": "this analysis is already running in this process"}, None]


def test_the_runner_names_itself_after_the_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    """It only has to differ from the other runners and stay stable while this
    process lives. In the cluster that is the pod name."""
    public.runner_id.cache_clear()
    monkeypatch.setenv("HOSTNAME", "marketmind-orchestrator-abc123")

    assert public.runner_id() == "marketmind-orchestrator-abc123"

    public.runner_id.cache_clear()


async def test_the_claim_is_renewed_on_a_clock_while_the_run_lasts(scoring: AsyncMock) -> None:
    """A lease is renewed on time, not on milestones. Renewing it only on
    progress reports was wrong: one collection step runs for minutes without
    reporting, and a claim that ages out mid-step tells another agent this
    analysis is orphaned while it is very much alive."""
    import asyncio

    graph = _graph()
    running = asyncio.Event()

    async def slow_run(_payload, config=None):
        running.set()
        await asyncio.sleep(3600)
        yield {}

    graph.astream = slow_run

    with (
        patch("app.graph.get_graph", return_value=graph),
        patch.object(public, "_CLAIM_RENEWAL_SECONDS", 0.01),
    ):
        run = asyncio.create_task(public._run("r1"))
        await running.wait()
        await asyncio.sleep(0.05)
        run.cancel()

    assert scoring.claim_analysis.await_count >= 2, "the lease has to keep being renewed"


async def test_the_claim_is_released_when_the_run_ends(scoring: AsyncMock) -> None:
    """A lease still being renewed after the work stopped is the one thing that
    would keep an orphan invisible."""
    import asyncio

    graph = _graph()

    with (
        patch("app.graph.get_graph", return_value=graph),
        patch.object(public, "_CLAIM_RENEWAL_SECONDS", 0.01),
    ):
        await public._run("r1")
        before = scoring.claim_analysis.await_count
        await asyncio.sleep(0.05)

    assert scoring.claim_analysis.await_count == before, "nothing renews after the run is over"


async def test_the_orphan_check_keeps_running_and_is_not_a_one_shot(scoring: AsyncMock) -> None:
    """Measured, not reasoned. A pod killed mid-analysis is replaced in about
    fifteen seconds, and the claim it held is good for two minutes — so the
    replacement starts, finds nothing orphaned, and if it never asks again the
    orphan waits for the fifteen-minute sweep. Which is the wait this was built
    to remove."""
    import asyncio

    scoring.orphaned_analyses = AsyncMock(return_value=[])

    with patch.object(public, "_ORPHAN_CHECK_SECONDS", 0.01):
        watch = asyncio.create_task(public.watch_for_orphans())
        await asyncio.sleep(0.06)
        watch.cancel()

    assert scoring.orphaned_analyses.await_count >= 3, "it has to keep asking"


async def test_the_watch_survives_a_pass_that_blows_up(scoring: AsyncMock) -> None:
    """It has to outlive any single failure: a watch that dies on one bad pass
    leaves every later orphan to the sweep, silently."""
    import asyncio

    scoring.orphaned_analyses = AsyncMock(side_effect=[RuntimeError("boom"), [], []])

    with patch.object(public, "_ORPHAN_CHECK_SECONDS", 0.01):
        watch = asyncio.create_task(public.watch_for_orphans())
        await asyncio.sleep(0.06)
        alive = not watch.done()
        watch.cancel()

    assert alive
    assert scoring.orphaned_analyses.await_count >= 3


# --- the narrative comes out in the language the buyer is reading in ---


def test_the_interface_language_is_registered_with_the_analysis(scoring: AsyncMock, started: list[str]) -> None:
    """The interface is EN/ES and the scoring service's default was PT: a reader
    who chose Spanish got a Portuguese explanation. The language is the buyer's
    choice, so it travels with the record and a resumed run reads it back."""
    _client().post("/analyze", json={"product_name": "air fryer", "countries": ["ESP"], "language": "ES"})

    assert scoring.register_analysis.await_args.args[0]["language"] == "es"


def test_no_language_asked_for_leaves_the_default_to_the_scoring_service(
    scoring: AsyncMock, started: list[str]
) -> None:
    _client().post("/analyze", json={"product_name": "air fryer", "countries": ["ESP"]})

    assert scoring.register_analysis.await_args.args[0]["language"] is None
