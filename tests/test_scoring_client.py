import httpx
import pytest
import respx

from app.scoring_client import ScoringClient, ScoringRejected, ScoringUnavailable

_BASE = "http://core.test"


def _client() -> ScoringClient:
    return ScoringClient(base_url=_BASE, token="secret")


@respx.mock
async def test_the_token_travels_on_every_call() -> None:
    """The tools write — they register analyses, collect, complete requests — so
    without the secret the service refuses them."""
    route = respx.post(f"{_BASE}/internal/analyses").mock(
        return_value=httpx.Response(201, json={"request_id": "r1", "status": "pending"})
    )

    await _client().register_analysis(countries=["ESP"])

    assert route.calls.last.request.headers["X-Internal-Token"] == "secret"


@respx.mock
async def test_no_token_sends_no_header_rather_than_an_empty_one() -> None:
    """An empty header reads like a configuration that was attempted. An absent
    one reads like one that was never made, which is the truth."""
    route = respx.post(f"{_BASE}/internal/analyses").mock(return_value=httpx.Response(201, json={}))

    await ScoringClient(base_url=_BASE, token=None).register_analysis(countries=["ESP"])

    assert "X-Internal-Token" not in route.calls.last.request.headers


# --- the two failures lead to opposite decisions ---


@respx.mock
async def test_an_unreachable_service_is_retryable() -> None:
    respx.post(f"{_BASE}/internal/analyses").mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(ScoringUnavailable):
        await _client().register_analysis(countries=["ESP"])


@respx.mock
async def test_a_server_error_is_retryable() -> None:
    respx.post(f"{_BASE}/internal/analyses").mock(return_value=httpx.Response(503, json={"detail": "down"}))

    with pytest.raises(ScoringUnavailable):
        await _client().register_analysis(countries=["ESP"])


@respx.mock
async def test_a_refusal_is_not_retryable_and_carries_the_reason() -> None:
    """Retrying a 4xx just repeats the answer. The reason is kept because it is
    what the analysis has to report when it gives up."""
    respx.post(f"{_BASE}/internal/coverage/ensure").mock(
        return_value=httpx.Response(401, json={"detail": "Invalid or missing X-Internal-Token."})
    )

    with pytest.raises(ScoringRejected) as excinfo:
        await _client().ensure_coverage("ESP", None)

    assert excinfo.value.status_code == 401
    assert "X-Internal-Token" in excinfo.value.detail


# --- the answers the agent decides on ---


@respx.mock
async def test_coverage_answers_with_the_tasks_it_dispatched() -> None:
    """The tasks are the answer, not the data: a FAILED one is a source that is
    down and worth insisting on, while a task that finished and left the
    indicator missing is a source that does not have that datum."""
    respx.post(f"{_BASE}/internal/coverage/ensure").mock(
        return_value=httpx.Response(
            200,
            json={
                "country": "ESP",
                "search_terms": ["freidora de aire", "air fryer"],
                "tasks": [{"status": "failed", "source": "undata", "error_message": "502"}],
            },
        )
    )

    answer = await _client().ensure_coverage("ESP", "e1")

    assert [t["status"] for t in answer["tasks"]] == ["failed"]


@respx.mock
async def test_a_cached_champion_comes_back_with_its_figure_not_just_the_fact() -> None:
    """The figure decides whether it is still a champion. A stored one that a
    requested market now beats has to be replaced, not crowned again."""
    respx.post(f"{_BASE}/internal/candidates/coverage").mock(
        return_value=httpx.Response(
            200, json={"entity_key": "e1", "champion_cached": True, "champion_country": "BRA", "champion_mos": 66.37}
        )
    )

    body = await _client().candidates_coverage("e1", ["ESP"], "2026-09")

    assert body["champion_cached"] is True


@respx.mock
async def test_the_demand_ceiling_comes_back_with_the_ruler_that_produced_it() -> None:
    respx.post(f"{_BASE}/internal/demand-ceiling").mock(
        return_value=httpx.Response(
            200, json={"demand_ceiling": 1_500_000.0, "demand_ruler": {"mode": "reference_candidate"}}
        )
    )

    ceiling, ruler = await _client().demand_ceiling("e1", ["ESP"], "2026-09")

    assert ceiling == 1_500_000.0
    assert ruler["mode"] == "reference_candidate"


@respx.mock
async def test_absent_evidence_comes_back_as_absent() -> None:
    """Not as an empty packet, which would read downstream as "nothing to say
    about this market" — a different claim."""
    respx.post(f"{_BASE}/internal/evidence").mock(return_value=httpx.Response(200, json={"packet": None}))

    assert await _client().build_evidence("e1", ["ESP"], "2026-09") is None


@respx.mock
async def test_a_failed_narration_comes_back_as_none() -> None:
    """Best-effort by design: the numbers are the answer, the prose is a reading
    of them, and a missing reading never invalidates the answer."""
    respx.post(f"{_BASE}/internal/narrate").mock(return_value=httpx.Response(200, json={"narrative": None}))

    assert await _client().narrate({"countries": []}, "air fryer") is None


@respx.mock
async def test_a_champion_can_be_recorded_as_nothing_scorable() -> None:
    route = respx.post(f"{_BASE}/internal/champion").mock(return_value=httpx.Response(200, json={}))

    await _client().set_champion("e1", "2026-09", country=None, mos=None)

    import json

    body = json.loads(route.calls.last.request.content)
    assert body["country"] is None
    assert body["period_key"] == "2026-09"


@respx.mock
async def test_what_a_call_spent_on_models_is_added_to_the_run() -> None:
    """The agent decides; the scoring service makes the model calls those
    decisions cause. Without this the only component able to say what a run cost
    would be the one component that cannot see it."""
    from app import spend

    respx.post(f"{_BASE}/internal/analyses/r1/enrich-terms").mock(
        return_value=httpx.Response(
            200,
            json={"ok": True},
            headers={"X-LLM-Input-Tokens": "1200", "X-LLM-Output-Tokens": "300", "X-LLM-Calls": "2"},
        )
    )

    with spend.tally() as spent:
        await _client()._call("POST", "/internal/analyses/r1/enrich-terms")

    assert spent.input_tokens == 1200
    assert spent.output_tokens == 300
    assert spent.calls == 2


@respx.mock
async def test_a_call_that_failed_still_reports_what_it_burned() -> None:
    """A run that retries three times has to report all three. Tokens spent
    before a failure are spent."""
    from app import spend

    respx.post(f"{_BASE}/internal/analyses/r1/enrich-terms").mock(
        return_value=httpx.Response(
            500,
            json={"detail": "upstream died"},
            headers={"X-LLM-Input-Tokens": "900", "X-LLM-Output-Tokens": "0", "X-LLM-Calls": "1"},
        )
    )

    with spend.tally() as spent:
        with pytest.raises(ScoringUnavailable):
            await _client()._call("POST", "/internal/analyses/r1/enrich-terms")

    assert spent.total_tokens == 900


def test_a_call_made_during_a_run_carries_the_runs_trace(monkeypatch) -> None:
    """So the model calls it causes in the scoring service land inside the run's
    trace, under the step that asked, instead of in traces of their own."""
    from langfuse import Langfuse

    from app import observability
    from app.scoring_client import ScoringClient

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    observability._handler.cache_clear()
    analysis_id = "a8c5d38b-db60-413c-b8f3-086a0befb83a"

    try:
        with observability.run(analysis_id):
            headers = ScoringClient(base_url="http://core", token="t").headers()
    finally:
        observability._handler.cache_clear()

    assert headers["X-Internal-Token"] == "t"
    assert headers["traceparent"].split("-")[1] == Langfuse.create_trace_id(seed=analysis_id)


def test_a_call_outside_a_run_carries_no_trace() -> None:
    from app.scoring_client import ScoringClient

    assert "traceparent" not in ScoringClient(base_url="http://core", token="t").headers()
