"""The five measures, one definition each."""

import logging
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app import metrics, spend


def _lines(caplog) -> str:
    return caplog.text


# --- task_completion_rate ---


def test_completion_counts_only_the_markets_that_were_asked_for(caplog) -> None:
    """A run adds the period's champion for comparison even when nobody asked
    for it. Counting that market would let a run that failed the actual request
    still report a rate above zero."""
    results = [
        {"country": "ESP", "mos": 40.0},
        {"country": "PRT", "mos": None},  # asked for, not scored
        {"country": "USA", "mos": 91.0},  # the champion, nobody asked
    ]
    with caplog.at_level(logging.INFO):
        metrics.task_completion_rate(results, {"ESP", "PRT"})

    assert "task_completion_rate=0.5 (1 of 2 requested markets scored)" in _lines(caplog)


def test_a_run_that_delivered_everything_reports_one(caplog) -> None:
    with caplog.at_level(logging.INFO):
        metrics.task_completion_rate([{"country": "ESP", "mos": 40.0}], {"ESP"})

    assert "task_completion_rate=1.0" in _lines(caplog)


# --- response_latency_seconds ---


def test_latency_is_the_wait_the_buyer_actually_had(caplog) -> None:
    started = (datetime.now(UTC) - timedelta(seconds=90)).isoformat()
    with caplog.at_level(logging.INFO):
        metrics.response_latency_seconds(started)

    line = _lines(caplog)
    assert "response_latency_seconds=9" in line  # ~90, wall clock
    assert "run start -> answer delivered" in line


def test_a_start_time_that_cannot_be_read_reports_nothing(caplog) -> None:
    """Rather than a zero, which would drag the average down with a figure that
    was never measured."""
    with caplog.at_level(logging.INFO):
        metrics.response_latency_seconds("not a timestamp")
        metrics.response_latency_seconds(None)

    assert "response_latency_seconds" not in _lines(caplog)


# --- token_consumption ---


def test_tokens_are_summed_across_every_call_the_run_made(caplog) -> None:
    with spend.tally():
        spend.add_response(
            httpx.Headers({"X-LLM-Input-Tokens": "1200", "X-LLM-Output-Tokens": "300", "X-LLM-Calls": "1"})
        )
        spend.add_response(
            httpx.Headers({"X-LLM-Input-Tokens": "800", "X-LLM-Output-Tokens": "100", "X-LLM-Calls": "2"})
        )
        spend.add_call(500, 50)  # the judge, called by this service itself
        with caplog.at_level(logging.INFO):
            metrics.token_consumption()

    line = _lines(caplog)
    assert "token_consumption=2950.0" in line
    assert "2500 in + 450 out over 4 model calls" in line


def test_a_response_that_reports_no_model_call_adds_nothing() -> None:
    with spend.tally() as spent:
        spend.add_response(httpx.Headers({"content-type": "application/json"}))

    assert spent.calls == 0
    assert spent.total_tokens == 0


def test_a_provider_that_reported_no_figures_is_visible_as_such() -> None:
    """A total of zero tokens over three calls has to read as 'not reported' and
    not as 'free', or the cost understates itself every time a gateway omits the
    counts."""
    with spend.tally() as spent:
        spend.add_response(httpx.Headers({"X-LLM-Calls": "3", "X-LLM-Calls-Without-Usage": "3"}))

    assert spent.calls == 3
    assert "3 of them reported no usage" in str(spent)


def test_one_run_never_sees_another_run_s_tokens() -> None:
    with spend.tally() as first:
        spend.add_call(100, 10)
    with spend.tally() as second:
        spend.add_call(5, 1)

    assert first.total_tokens == 110
    assert second.total_tokens == 6


# --- tool_invocation_accuracy ---


def test_the_same_country_collected_twice_costs_accuracy(caplog) -> None:
    tasks = [{"country": "ESP"}, {"country": "ESP"}, {"country": "PRT"}]
    with caplog.at_level(logging.INFO):
        metrics.tool_invocation_accuracy(tasks, ["ESP", "PRT"])

    line = _lines(caplog)
    assert "tool_invocation_accuracy=0.6667" in line
    assert "1 redundant, 0 off-request" in line


def test_a_country_nobody_asked_about_costs_accuracy(caplog) -> None:
    tasks = [{"country": "ESP"}, {"country": "FRA"}]
    with caplog.at_level(logging.INFO):
        metrics.tool_invocation_accuracy(tasks, ["ESP"])

    assert "0 redundant, 1 off-request" in _lines(caplog)


def test_a_run_that_collected_nothing_says_so_rather_than_vanishing(caplog) -> None:
    with caplog.at_level(logging.INFO):
        metrics.tool_invocation_accuracy([], ["ESP"])

    line = _lines(caplog)
    assert "tool_invocation_accuracy=1.0" in line
    assert "no collection call was needed" in line, "the 1.0 must not read as a score over real calls"


# --- hallucination_rate ---


def test_a_figure_the_evidence_does_not_hold_is_a_hallucination(caplog) -> None:
    packet = {"countries": [{"country": "ESP", "mos": 40.02}]}
    narrative = "Spain scores 40 with 9,400,000 searches a month."
    with caplog.at_level(logging.INFO):
        metrics.hallucination_rate(narrative, packet, [9_400_000.0])

    line = _lines(caplog)
    assert "hallucination_rate=0.5" in line  # 1 of the 2 figures stated
    assert "9400000" in line


def test_a_narrative_written_from_the_evidence_scores_zero(caplog) -> None:
    packet = {"countries": [{"country": "ESP", "mos": 40.02}]}
    with caplog.at_level(logging.INFO):
        metrics.hallucination_rate("Spain scores 40.", packet, [])

    assert "hallucination_rate=0.0 (0 of 1 stated figures" in _lines(caplog)


def test_a_narrative_with_no_figure_says_there_was_nothing_to_check(caplog) -> None:
    """Not 'clean'. A text that states nothing measurable was not verified, and
    a rate of zero over an empty denominator has to say which of the two it is."""
    with caplog.at_level(logging.INFO):
        metrics.hallucination_rate("Spain looks promising.", {}, [])

    assert "nothing to check" in _lines(caplog)


@pytest.mark.parametrize("narrative", [None, ""])
def test_no_narrative_at_all_is_not_reported_as_truthful(narrative, caplog) -> None:
    with caplog.at_level(logging.INFO):
        metrics.hallucination_rate(narrative, {}, [])

    assert "nothing to check" in _lines(caplog)


async def test_two_runs_at_once_do_not_share_a_tally() -> None:
    """The tally is a context variable, and the run is an async generator driven
    by a per-request task. Asserted rather than reasoned about: if the scope were
    wrong, one analysis would report another analysis's cost, and the figure
    would be wrong in a way nobody would notice."""
    import asyncio

    async def run(calls: int, tokens: int) -> int:
        with spend.tally() as spent:
            for _ in range(calls):
                spend.add_call(tokens, 0)
                await asyncio.sleep(0)  # let the other run interleave here
            return spent.total_tokens

    first, second = await asyncio.gather(run(3, 100), run(2, 7))

    assert first == 300
    assert second == 14
