"""Long-term memory: which markets the sources do not hold data for.

With InMemoryStore throughout, so nothing here needs a database -- the same
guarantee the design leans on, that CHECKPOINTER=memory and CHECKPOINTER=postgres
behave alike because both are the same store interface underneath.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from langgraph.store.memory import InMemoryStore

from app import nodes
from app.graph import build_graph

_NAMESPACE = ("markets", "barren")


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    client = AsyncMock()
    monkeypatch.setattr(nodes, "_client", lambda: client)
    return client


def _state(**overrides) -> dict:
    base = {
        "analysis_id": "a1",
        "entity_key": "e1",
        "countries": ["ESP", "PRT"],
        "period_key": "2026-09",
        "attempts": 1,
        "coverage_tasks": [],
        "mos_results": [],
        "deadline_at": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
    }
    base.update(overrides)
    return base


async def _seed(store, entity_key: str, country: str, *, days_old: float = 0.0, analysis_id: str = "a0") -> datetime:
    """Write a barren record directly, the way an earlier run's give_up would."""
    recorded_at = datetime.now(UTC) - timedelta(days=days_old)
    await store.aput(
        _NAMESPACE,
        f"{entity_key}:{country}",
        {"recorded_at": recorded_at.isoformat(), "analysis_id": analysis_id, "cause": nodes._NO_DATA_CAUSE},
    )
    return recorded_at


def _called_countries(mock: AsyncMock) -> list[str]:
    return [call.args[0] for call in mock.await_args_list]


# --- writing: give_up_node remembers only a permanent "no data" ---


async def test_a_barren_giveup_writes_one_record_per_missing_market(core: AsyncMock) -> None:
    store = InMemoryStore()
    state = _state(
        countries=["ESP", "PRT", "BRA"],
        mos_results=[{"country": "ESP", "mos": 70.0}],
        coverage_tasks=[{"country": "PRT", "status": "success"}, {"country": "BRA", "status": "success"}],
    )

    await nodes.give_up_node(state, store=store)

    prt = await store.aget(_NAMESPACE, "e1:PRT")
    bra = await store.aget(_NAMESPACE, "e1:BRA")
    assert prt is not None and bra is not None
    assert prt.value["analysis_id"] == "a1"
    assert prt.value["cause"] == "the sources do not hold the data for these markets"
    assert "recorded_at" in prt.value
    # ESP scored, so it was never missing and gets no record at all.
    assert await store.aget(_NAMESPACE, "e1:ESP") is None


async def test_an_outage_giveup_writes_nothing(core: AsyncMock) -> None:
    """A source being down is about the world that day, not the product."""
    store = InMemoryStore()
    state = _state(
        mos_results=[{"country": "ESP", "mos": 70.0}],
        coverage_tasks=[{"country": "PRT", "status": "failed", "error_message": "502"}],
    )

    await nodes.give_up_node(state, store=store)

    assert await store.aget(_NAMESPACE, "e1:PRT") is None


async def test_a_deadline_giveup_writes_nothing(core: AsyncMock) -> None:
    """Running out of time says nothing about whether the sources have the data."""
    store = InMemoryStore()
    state = _state(
        mos_results=[{"country": "ESP", "mos": 70.0}],
        coverage_tasks=[{"country": "PRT", "status": "success"}],
        deadline_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
    )

    await nodes.give_up_node(state, store=store)

    assert await store.aget(_NAMESPACE, "e1:PRT") is None


async def test_a_recalled_market_is_not_rewritten(core: AsyncMock) -> None:
    """It was not re-verified this run, so rewriting it would reset a TTL that
    exists precisely to force a real recheck eventually."""
    store = InMemoryStore()
    recorded_at = await _seed(store, "e1", "PRT")
    state = _state(
        mos_results=[{"country": "ESP", "mos": 70.0}],
        coverage_tasks=[],
        recalled_countries={"PRT": recorded_at.isoformat()},
    )

    await nodes.give_up_node(state, store=store)

    stored = await store.aget(_NAMESPACE, "e1:PRT")
    assert stored.value["recorded_at"] == recorded_at.isoformat(), "untouched, not refreshed"
    assert stored.value["analysis_id"] == "a0", "still the run that actually discovered it"


async def test_the_recalled_giveup_reason_names_the_date(core: AsyncMock) -> None:
    store = InMemoryStore()
    recorded_at = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)
    state = _state(
        mos_results=[{"country": "ESP", "mos": 70.0}],
        coverage_tasks=[],
        recalled_countries={"PRT": recorded_at.isoformat()},
    )

    result = await nodes.give_up_node(state, store=store)

    reason = result["errors"][0]
    assert "no data when this was last checked, on 2026-08-24" in reason


# --- reading: collect_node checks memory before calling the scoring client ---


async def test_a_fresh_record_short_circuits_collection(core: AsyncMock) -> None:
    store = InMemoryStore()
    recorded_at = await _seed(store, "e1", "PRT")
    core.ensure_coverage = AsyncMock(return_value={"tasks": []})

    result = await nodes.collect_node(_state(), store=store)

    assert _called_countries(core.ensure_coverage) == ["ESP"], "PRT is answered from memory, never collected"
    assert result["recalled_countries"] == {"PRT": recorded_at.isoformat()}
    assert result["countries_collected"] == ["ESP"]


async def test_a_stale_record_is_ignored_collected_and_deleted(core: AsyncMock) -> None:
    store = InMemoryStore()
    await _seed(store, "e1", "PRT", days_old=31)
    core.ensure_coverage = AsyncMock(return_value={"tasks": []})

    result = await nodes.collect_node(_state(), store=store)

    assert _called_countries(core.ensure_coverage) == ["ESP", "PRT"], "stale, so tried again like any other country"
    assert result["recalled_countries"] == {}
    assert await store.aget(_NAMESPACE, "e1:PRT") is None


async def test_a_record_just_inside_the_ttl_is_still_trusted(core: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_TTL_DAYS", "10")
    store = InMemoryStore()
    await _seed(store, "e1", "PRT", days_old=9)
    core.ensure_coverage = AsyncMock(return_value={"tasks": []})

    result = await nodes.collect_node(_state(), store=store)

    assert _called_countries(core.ensure_coverage) == ["ESP"]
    assert "PRT" in result["recalled_countries"]


# --- forgetting: score_node deletes the record of a market that scores ---


async def test_a_scored_market_has_its_record_deleted(core: AsyncMock) -> None:
    store = InMemoryStore()
    await _seed(store, "e1", "PRT")
    core.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}])

    await nodes.score_node(_state(), store=store)

    assert await store.aget(_NAMESPACE, "e1:PRT") is None


async def test_an_unscored_market_keeps_its_record(core: AsyncMock) -> None:
    store = InMemoryStore()
    await _seed(store, "e1", "PRT")
    core.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}])

    await nodes.score_node(_state(), store=store)

    assert await store.aget(_NAMESPACE, "e1:PRT") is not None


# --- no store configured: every path behaves exactly as today ---


async def test_with_no_store_collection_is_unaffected(core: AsyncMock) -> None:
    core.ensure_coverage = AsyncMock(return_value={"tasks": []})

    result = await nodes.collect_node(_state())

    assert _called_countries(core.ensure_coverage) == ["ESP", "PRT"]
    assert result["recalled_countries"] == {}


async def test_with_no_store_scoring_is_unaffected(core: AsyncMock) -> None:
    core.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 70.0}])

    result = await nodes.score_node(_state())

    assert result["mos_results"] == [{"country": "ESP", "mos": 70.0}]


async def test_with_no_store_giveup_is_unaffected(core: AsyncMock) -> None:
    state = _state(mos_results=[], coverage_tasks=[{"country": "PRT", "status": "success"}])

    result = await nodes.give_up_node(state)

    assert "do not hold the data" in result["errors"][0]


# --- memory never fails a run ---


class _ExplodingStore:
    """Every call raises, the way a store with a dead connection would."""

    async def aget(self, *args, **kwargs):
        raise RuntimeError("store is down")

    async def aput(self, *args, **kwargs):
        raise RuntimeError("store is down")

    async def adelete(self, *args, **kwargs):
        raise RuntimeError("store is down")


async def test_a_broken_store_does_not_stop_collection(core: AsyncMock, caplog: pytest.LogCaptureFixture) -> None:
    core.ensure_coverage = AsyncMock(return_value={"tasks": []})

    with caplog.at_level("DEBUG"):
        result = await nodes.collect_node(_state(), store=_ExplodingStore())

    assert result["countries_collected"] == ["ESP", "PRT"]
    assert "Could not read the market memory" in caplog.text


async def test_a_broken_store_does_not_stop_scoring(core: AsyncMock) -> None:
    core.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 70.0}])

    result = await nodes.score_node(_state(), store=_ExplodingStore())

    assert result["mos_results"] == [{"country": "ESP", "mos": 70.0}]


async def test_a_broken_store_does_not_stop_giving_up(core: AsyncMock) -> None:
    state = _state(mos_results=[], coverage_tasks=[{"country": "PRT", "status": "success"}])

    result = await nodes.give_up_node(state, store=_ExplodingStore())

    assert "do not hold the data" in result["errors"][0]


# --- recall_hits, through the whole graph ---


@pytest.fixture
def full_run(monkeypatch: pytest.MonkeyPatch, core: AsyncMock) -> AsyncMock:
    """The fixture from test_graph.py's `core`, repeated here rather than
    imported: this file wants to seed the store between read_analysis and the
    first collection, which the shared fixture has no hook for."""
    core.read_analysis = AsyncMock(
        return_value={"entity_key": "e1", "canonical_title": "air fryer", "countries": ["ESP", "PRT"], "user_prompt": None}
    )
    core.candidates_coverage = AsyncMock(return_value={"champion_cached": False})
    core.demand_ceiling = AsyncMock(return_value=(1_500_000.0, {"mode": "reference_candidate"}))
    core.score_candidates = AsyncMock(return_value=[])
    core.build_evidence = AsyncMock(return_value={"countries": [{"country": "ESP"}, {"country": "PRT"}]})
    core.narrate = AsyncMock(return_value="Spain leads.")
    return core


async def test_recall_hits_is_zero_when_nothing_was_recalled(full_run: AsyncMock, caplog: pytest.LogCaptureFixture) -> None:
    full_run.ensure_coverage = AsyncMock(return_value={"tasks": []})
    full_run.compute_mos = AsyncMock(
        return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}]
    )
    store = InMemoryStore()

    with caplog.at_level("INFO"):
        await build_graph(store=store).ainvoke({"analysis_id": "a1"})

    assert "QUALITY recall_hits=0.0" in caplog.text


async def test_recall_hits_counts_the_markets_answered_from_memory(
    full_run: AsyncMock, caplog: pytest.LogCaptureFixture
) -> None:
    store = InMemoryStore()
    recorded_at = await _seed(store, "e1", "PRT")
    full_run.ensure_coverage = AsyncMock(return_value={"tasks": []})
    # PRT is never collected (recalled), so nothing produces a score for it --
    # exactly what a real run would see from a source it never asked again.
    full_run.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}])

    with caplog.at_level("INFO"):
        final = await build_graph(store=store).ainvoke({"analysis_id": "a1"})

    assert _called_countries(full_run.ensure_coverage) == ["ESP"]
    assert "QUALITY recall_hits=1.0" in caplog.text
    reason = final["errors"][0]
    assert f"no data when this was last checked, on {recorded_at.date().isoformat()}" in reason
    full_run.fail_analysis.assert_awaited_once()
    full_run.complete_analysis.assert_not_awaited()


async def test_a_declared_market_is_never_sent_to_the_scoring_client_for_collection(
    full_run: AsyncMock,
) -> None:
    """The literal claim from the spec: a fresh memory means the collection
    client itself is never called for that market."""
    store = InMemoryStore()
    await _seed(store, "e1", "PRT")
    full_run.ensure_coverage = AsyncMock(return_value={"tasks": []})
    full_run.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}])

    await build_graph(store=store).ainvoke({"analysis_id": "a1"})

    full_run.ensure_coverage.assert_awaited_once_with("ESP", "e1")
