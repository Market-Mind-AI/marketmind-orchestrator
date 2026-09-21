"""The five measures every run reports, and what each one actually means.

    task_completion_rate       did it deliver what was asked for
    response_latency_seconds   how long the buyer waited for it
    token_consumption          what it cost to produce
    tool_invocation_accuracy   how much of the work was work worth doing
    hallucination_rate         how much of the text is not held up by evidence

They are computed here rather than at the call sites so that each has ONE
definition. A rate whose denominator changes depending on which node emitted it
is not a measurement of anything.

Each is logged as a QUALITY line and attached to the run's trace, so a single
bad run can be read in the log and a trend can be read in Langfuse. None of
them can fail a run: they are read after the answer is already delivered.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from app import observability, quality, spend

logger = logging.getLogger(__name__)

TASK_COMPLETION_RATE = "task_completion_rate"
RESPONSE_LATENCY_SECONDS = "response_latency_seconds"
TOKEN_CONSUMPTION = "token_consumption"
TOOL_INVOCATION_ACCURACY = "tool_invocation_accuracy"
HALLUCINATION_RATE = "hallucination_rate"


def task_completion_rate(results: list[dict[str, Any]], requested: set[str]) -> None:
    """Of the markets the buyer asked about, how many came back with a score.

    The requested ones only. A run adds the period's champion for comparison
    even when nobody asked for it, and counting that market would let a run
    that failed the actual request still report a rate above zero.
    """
    if not requested:
        return
    scored = len([r for r in results if r.get("mos") is not None and r.get("country") in requested])
    observability.record(
        TASK_COMPLETION_RATE,
        round(scored / len(requested), 4),
        comment=f"{scored} of {len(requested)} requested markets scored",
    )


def response_latency_seconds(started_at: str | None, *, what: str = "answer delivered") -> None:
    """From the moment the run started to the moment the buyer has the answer.

    Wall clock, not work time, because that is what was actually waited. A run
    that resumed after a restart reports the whole wait including the gap, which
    is the honest figure: the buyer waited through it.
    """
    started = _parse(started_at)
    if started is None:
        return
    observability.record(
        RESPONSE_LATENCY_SECONDS,
        round((datetime.now(UTC) - started).total_seconds(), 2),
        comment=f"run start -> {what}",
    )


def token_consumption() -> None:
    """Everything the run spent on models, wherever the call was made.

    Mostly in the scoring service, which reports each response's spend on its
    headers; the judge is the one model this service calls itself.
    """
    spent = spend.current()
    if not spent.calls:
        observability.record(TOKEN_CONSUMPTION, 0.0, comment="no model call in this run")
        return
    observability.record(TOKEN_CONSUMPTION, float(spent.total_tokens), comment=str(spent))


def tool_invocation_accuracy(coverage_tasks: list[dict[str, Any]], countries: list[str]) -> None:
    """The share of collection calls that were worth making.

    Two ways to get it wrong, and they are counted together here but named
    apart in the comment, because they are fixed in opposite places: a
    redundant call is the same country collected twice, which is where cost
    leaks; an off-request call is a country nobody asked about, which is a tool
    invoked on the wrong argument.
    """
    total = len(coverage_tasks)
    if not total:
        # A run whose countries were all still fresh collects nothing, and made
        # no wrong call. Reported rather than omitted, with the denominator said
        # out loud so the 1.0 is not read as a perfect score over real calls.
        observability.record(
            TOOL_INVOCATION_ACCURACY,
            1.0,
            comment="no collection call was needed; every market was already fresh",
        )
        return
    faults = quality.tool_call_faults(coverage_tasks, countries)
    wrong = faults["redundant"] + faults["off_request"]
    observability.record(
        TOOL_INVOCATION_ACCURACY,
        round((total - wrong) / total, 4),
        comment=(
            f"{total - wrong} of {total} collection calls were needed and on-request "
            f"({faults['redundant']} redundant, {faults['off_request']} off-request)"
        ),
    )


def hallucination_rate(narrative: str | None, packet: dict[str, Any], unsupported: list[float]) -> None:
    """The share of the narrative's figures that the evidence does not hold.

    Measurable exactly, and only because no number in an answer comes from a
    model: the text is written from a deterministic packet, so a figure that is
    not in the packet is not supported by anything. The denominator is the
    figures the text states, which is what "rate" has to mean here — a
    narrative that says nothing measurable has nothing to hallucinate and is
    reported as such rather than as clean.

    The judge covers what this cannot reach — claims made without a figure —
    and is recorded separately, because a sampled signal and an exact one must
    not be averaged into a single number.
    """
    stated = quality.numbers_in(narrative or "")
    if not stated:
        observability.record(
            HALLUCINATION_RATE,
            0.0,
            comment="the narrative states no figure, so there is nothing to check",
        )
        return
    observability.record(
        HALLUCINATION_RATE,
        round(len(unsupported) / len(stated), 4),
        comment=(
            f"{len(unsupported)} of {len(stated)} stated figures are not in the evidence"
            + (f": {unsupported}" if unsupported else "")
        ),
    )


def _parse(when: str | None) -> datetime | None:
    if not when:
        return None
    try:
        parsed = datetime.fromisoformat(when)
    except ValueError:
        logger.debug("Could not read the run's start time from %r", when)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
