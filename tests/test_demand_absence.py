"""Markets with no recent demand are declared, not collected again.

The scoring service answers which ones from its own data, per product and
brand. What used to be a memory kept here, keyed by product alone and trusted
for a window of its own, is gone.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from app import nodes
from app.scoring_client import ScoringUnavailable


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    client = AsyncMock()
    client.ensure_coverage = AsyncMock(return_value={"tasks": []})
    client.demand_absence = AsyncMock(return_value={"markets": [], "window_days": 30})
    monkeypatch.setattr(nodes, "_client", lambda: client)
    return client


def _state(**overrides) -> dict:
    base = {
        "analysis_id": "a1",
        "entity_key": "e1",
        "brand_key": "b1",
        "countries": ["ESP", "PRT"],
        "period_key": "2026-09",
        "attempts": 1,
        "coverage_tasks": [],
        "mos_results": [],
        "deadline_at": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
    }
    base.update(overrides)
    return base


async def test_the_question_is_asked_per_product_brand_and_period(core: AsyncMock) -> None:
    await nodes.collect_node(_state())

    core.demand_absence.assert_awaited_once_with("e1", ["ESP", "PRT"], "2026-09", "b1")


async def test_an_absent_market_is_not_collected(core: AsyncMock) -> None:
    core.demand_absence = AsyncMock(
        return_value={
            "markets": [
                {"country": "PRT", "absent": True, "checked_at": "2026-09-14T10:00:00+00:00"},
                {"country": "ESP", "absent": False, "checked_at": None},
            ],
            "window_days": 30,
        }
    )

    answer = await nodes.collect_node(_state())

    assert [call.args[0] for call in core.ensure_coverage.await_args_list] == ["ESP"]
    assert answer["absent_countries"] == {"PRT": "2026-09-14T10:00:00+00:00"}


async def test_not_knowing_is_not_absent(core: AsyncMock) -> None:
    """A market with a term still to collect answers absent=false, and is
    collected like any other."""
    core.demand_absence = AsyncMock(
        return_value={"markets": [{"country": "PRT", "absent": False, "checked_at": None}], "window_days": 30}
    )

    await nodes.collect_node(_state())

    assert core.ensure_coverage.await_count == 2


async def test_an_unanswered_question_collects_everything(core: AsyncMock) -> None:
    """The skip is an optimisation. A scoring service that cannot answer it
    leaves the run collecting, as it did before it could ask."""
    core.demand_absence = AsyncMock(side_effect=ScoringUnavailable("down"))

    answer = await nodes.collect_node(_state())

    assert core.ensure_coverage.await_count == 2
    assert answer["absent_countries"] == {}


async def test_the_giveup_reason_names_when_the_demand_was_last_checked(core: AsyncMock) -> None:
    state = _state(
        mos_results=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}],
        absent_countries={"PRT": "2026-09-14T10:00:00+00:00"},
    )

    await nodes.give_up_node(state)

    reason = core.fail_analysis.await_args.args[1]
    assert "PRT" in reason
    assert "2026-09-14" in reason


async def test_a_market_found_empty_this_run_is_said_plainly(core: AsyncMock) -> None:
    state = _state(
        mos_results=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}],
        coverage_tasks=[{"country": "PRT", "status": "success"}],
    )

    await nodes.give_up_node(state)

    assert nodes._NO_DATA_CAUSE in core.fail_analysis.await_args.args[1]


async def test_giving_up_flags_the_trace_as_an_error_with_the_reason(core: AsyncMock, monkeypatch) -> None:
    """A desistência termina a execução limpamente, então o trace listava como
    saudável (level DEFAULT) — e a view Quality do Langfuse ficava vazia
    enquanto análises falhavam (19/10/2026)."""
    flags: list[tuple[str, str]] = []
    monkeypatch.setattr(nodes.observability, "flag", lambda level, message: flags.append((level, message)))
    state = _state(
        mos_results=[{"country": "ESP", "mos": 70.0}],
        coverage_tasks=[{"country": "PRT", "status": "failed", "error_message": "502"}],
        attempts=3,
    )

    await nodes.give_up_node(state)

    assert flags and flags[0][0] == "ERROR"
    assert "PRT" in flags[0][1]
