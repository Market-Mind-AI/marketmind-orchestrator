"""The comparison itself, since it is what the migration is judged on."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.parity_check import compare  # noqa: E402


def _body(status: str = "done", result=None, breakdown=None) -> dict:
    return {"status": status, "result": result or [], "breakdown": breakdown}


def _row(country: str, mos: float | None = 70.0, **extra) -> dict:
    row = {
        "country": country,
        "mos": mos,
        "mos_before_cap": mos,
        "capped_by": None,
        "topsis_score": 0.7,
        "mos_interpretation": "Good Opportunity",
        "availability_status": "available",
        "excluded_reason": None,
    }
    row.update(extra)
    return row


def test_identical_answers_have_nothing_to_report() -> None:
    body = _body(result=[_row("ESP"), _row("PRT", 61.0)])

    assert compare(body, body) == []


def test_a_different_score_is_caught() -> None:
    legacy = _body(result=[_row("ESP", 70.0)])
    agent = _body(result=[_row("ESP", 70.01)])

    assert any("ESP.mos" in p for p in compare(legacy, agent))


def test_a_market_present_in_only_one_answer_is_caught() -> None:
    """The worst kind of difference: it changes the ranking rather than a
    figure in it."""
    legacy = _body(result=[_row("ESP"), _row("PRT")])
    agent = _body(result=[_row("ESP")])

    problems = compare(legacy, agent)

    assert any("PRT" in p and "missing from the agent" in p for p in problems)


def test_an_extra_market_in_the_agent_answer_is_caught() -> None:
    legacy = _body(result=[_row("ESP")])
    agent = _body(result=[_row("ESP"), _row("BRA", 88.0)])

    assert any("BRA" in p for p in compare(legacy, agent))


def test_a_country_excluded_for_a_different_reason_is_caught() -> None:
    legacy = _body(result=[_row("PER", None, excluded_reason="Missing mandatory indicator(s): trend_score")])
    agent = _body(result=[_row("PER", None, excluded_reason="No calibrated weight for dimension=Demand")])

    assert any("PER.excluded_reason" in p for p in compare(legacy, agent))


def test_the_same_score_from_different_dimensions_is_still_a_difference() -> None:
    """Two identical MOS values built from different subscores would be a
    coincidence, not agreement."""
    dims = lambda demand: {  # noqa: E731
        "countries": [{"country": "ESP", "dimensions": {"Demand": {"subscore": demand}, "Market Size": {"subscore": 50.0}}}]
    }
    legacy = _body(result=[_row("ESP")], breakdown=dims(80.0))
    agent = _body(result=[_row("ESP")], breakdown=dims(60.0))

    assert any("ESP.dimensions.Demand" in p for p in compare(legacy, agent))


def test_a_failed_run_against_a_done_one_is_caught() -> None:
    legacy = _body(status="done", result=[_row("ESP")])
    agent = _body(status="failed")

    problems = compare(legacy, agent)

    assert any("status" in p for p in problems)


def test_the_narrative_is_not_compared() -> None:
    """It is a model's prose about identical numbers: comparing it would fail
    on wording and say nothing about the method."""
    legacy = {**_body(result=[_row("ESP")]), "explanation": "Spain leads the ranking."}
    agent = {**_body(result=[_row("ESP")]), "explanation": "Spain comes first."}

    assert compare(legacy, agent) == []


def test_ids_and_timestamps_are_not_compared() -> None:
    """They differ between two runs by definition."""
    legacy = _body(result=[_row("ESP", id="1", computed_at="2026-09-21T10:00:00")])
    agent = _body(result=[_row("ESP", id="2", computed_at="2026-09-21T11:00:00")])

    assert compare(legacy, agent) == []
