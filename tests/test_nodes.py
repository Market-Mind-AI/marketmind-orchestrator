"""The decision the analysis turns on: insist, give up, or deliver."""

from datetime import UTC, datetime, timedelta

import pytest

from app import nodes

_REQUESTED = ["ESP", "PRT", "BRA"]


def _state(**overrides) -> dict:
    base = {
        "analysis_id": "a1",
        "entity_key": "e1",
        "countries": list(_REQUESTED),
        "period_key": "2026-09",
        "attempts": 1,
        "coverage_tasks": [],
        "mos_results": [],
        "deadline_at": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
    }
    base.update(overrides)
    return base


def _scored(*countries: str) -> list[dict]:
    return [{"country": c, "mos": 70.0} for c in countries]


# --- what counts as done ---


def test_every_requested_market_scored_is_complete() -> None:
    assert nodes.after_scoring(_state(mos_results=_scored(*_REQUESTED))) == "complete"


def test_a_row_without_a_score_is_the_same_gap_as_no_row() -> None:
    """A country measured but unrankable is not a country delivered."""
    results = [*_scored("ESP", "PRT"), {"country": "BRA", "mos": None}]

    assert nodes.incomplete_countries(_state(mos_results=results)) == ["BRA"]


def test_an_off_request_market_does_not_make_the_request_complete() -> None:
    """The champion is a benchmark, not a substitute for what was asked."""
    results = [*_scored("ESP", "PRT"), {"country": "JPN", "mos": 80.0}]

    assert nodes.incomplete_countries(_state(mos_results=results)) == ["BRA"]


# --- insisting, and when to stop ---


def test_a_source_that_is_down_is_worth_insisting_on() -> None:
    """The August outage: nine of twenty-four countries dropped out of a ranking
    because a source was unreachable for twenty-nine minutes."""
    state = _state(
        mos_results=_scored("ESP", "PRT"),
        coverage_tasks=[{"country": "BRA", "status": "failed", "error_message": "502"}],
    )

    assert nodes.after_scoring(state) == "insist"


def test_a_source_that_simply_lacks_the_datum_is_not() -> None:
    """A collection that finished without failing and still left the indicator
    missing will produce the same nothing however many times it is asked."""
    state = _state(
        mos_results=_scored("ESP", "PRT"),
        coverage_tasks=[{"country": "BRA", "status": "success"}],
    )

    assert nodes.after_scoring(state) == "give_up"


def test_insisting_stops_at_the_attempt_limit() -> None:
    state = _state(
        attempts=nodes.MAX_ATTEMPTS,
        mos_results=_scored("ESP", "PRT"),
        coverage_tasks=[{"country": "BRA", "status": "failed"}],
    )

    assert nodes.after_scoring(state) == "give_up"


def test_insisting_stops_at_the_deadline() -> None:
    """Past the window the run fails with a reason rather than delivering part
    of a comparison."""
    state = _state(
        deadline_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        mos_results=_scored("ESP", "PRT"),
        coverage_tasks=[{"country": "BRA", "status": "failed"}],
    )

    assert nodes.after_scoring(state) == "give_up"


def test_a_complete_run_never_gives_up_even_past_the_deadline() -> None:
    """The deadline bounds insisting, not delivering. Work already finished is
    not thrown away because the clock ran out while it finished."""
    state = _state(
        deadline_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        mos_results=_scored(*_REQUESTED),
    )

    assert nodes.after_scoring(state) == "complete"


# --- the period is decided once ---


def test_the_period_is_the_calendar_month_in_utc() -> None:
    assert nodes._period_key(datetime(2026, 9, 21, 23, 59, tzinfo=UTC)) == "2026-09"
    assert nodes._period_key(datetime(2026, 1, 1, 0, 0, tzinfo=UTC)) == "2026-01"


async def test_a_resumed_run_keeps_the_period_it_started_with(monkeypatch: pytest.MonkeyPatch) -> None:
    """An analysis that begins in September and resumes in October must not
    silently change which month it is measuring."""
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.read_analysis = AsyncMock(return_value={"entity_key": "e1", "countries": ["ESP"]})
    monkeypatch.setattr(nodes, "_client", lambda: client)

    resumed = await nodes.start_node({"analysis_id": "a1", "period_key": "2026-09", "attempts": 2})

    assert resumed["period_key"] == "2026-09"
    assert resumed["attempts"] == 2


async def test_a_fresh_run_decides_the_period_and_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.read_analysis = AsyncMock(return_value={"entity_key": "e1", "countries": ["ESP"]})
    monkeypatch.setattr(nodes, "_client", lambda: client)

    started = await nodes.start_node({"analysis_id": "a1"})

    assert started["period_key"] == nodes._period_key()
    assert started["deadline_at"] is not None
    assert started["attempts"] == 0
