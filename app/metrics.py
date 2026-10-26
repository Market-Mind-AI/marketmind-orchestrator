"""The seven measures every run reports, and what each one actually means.

    task_completion_rate       did it deliver what was asked for
    response_latency_seconds   how long the buyer waited for it
    token_consumption          how many tokens it took to produce
    dataforseo_cost_usd        what the demand data cost, as DataForSEO billed it
    analysis_cost_usd          what the whole run cost, models and data together
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
DATAFORSEO_COST_USD = "dataforseo_cost_usd"
ANALYSIS_COST_USD = "analysis_cost_usd"
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

    All of it in the scoring service, which reports each response's spend on
    its headers; this service calls no model itself.
    """
    spent = spend.current()
    if not spent.calls:
        observability.record(TOKEN_CONSUMPTION, 0.0, comment="no model call in this run")
        return
    observability.record(TOKEN_CONSUMPTION, float(spent.total_tokens), comment=str(spent))


def demand_cost() -> None:
    """What the run's demand data cost at DataForSEO, in USD.

    The one cost of an analysis that is not a model, and the one the token
    figure cannot see. Each Google Ads task is billed by DataForSEO and the
    collector reports the amount; the scoring service relays it on the same
    headers as the tokens, and each collection already sits in the trace as
    an observation with that cost (the scoring service writes it there, next
    to the call that spent it). This is the run's total, as a score, so runs
    compare on it the way they compare on tokens.
    """
    spent = spend.current()
    if not spent.dataforseo_calls:
        observability.record(DATAFORSEO_COST_USD, 0.0, comment="no Google Ads collection in this run")
        return
    comment = f"{spent.dataforseo_calls} DataForSEO tasks"
    if spent.dataforseo_calls_without_cost:
        comment += f"; {spent.dataforseo_calls_without_cost} of them reported no cost"
    observability.record(DATAFORSEO_COST_USD, round(spent.dataforseo_usd, 6), comment=comment)


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

    The Langfuse evaluator covers what this cannot reach — claims made without
    a figure — and is recorded there, because a model's verdict and an exact
    check must not be averaged into a single number.
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


def analysis_cost(analysis_id: str) -> None:
    """Everything the run paid for, in one line and one score.

    Models in dollars (priced by the table Langfuse holds, in the service
    that made each call) plus DataForSEO, which the token figure cannot see.
    The one number someone asks for — "what did this analysis cost?" —
    written where it can be read without opening Langfuse:

        COST analysis=<id> total_usd=1.64 models_usd=0.56 (claude-opus-5 0.50, gpt-5.1 0.05)
            dataforseo_usd=1.08 (12 tasks)

    A call that could not be priced is said, not zeroed: the total is then a
    floor, and the line says so.
    """
    spent = spend.current()
    breakdown = ", ".join(f"{m} {usd:.4f}" for m, usd in sorted(spent.models_usd.items(), key=lambda kv: -kv[1]))
    line = (
        f"COST analysis={analysis_id} total_usd={spent.total_usd:.4f} "
        f"models_usd={spent.llm_usd:.4f} ({breakdown or 'no priced model call'}) "
        f"dataforseo_usd={spent.dataforseo_usd:.4f} ({spent.dataforseo_calls} tasks)"
    )
    unpriced = spent.llm_calls_without_cost + spent.dataforseo_calls_without_cost
    if unpriced:
        line += f" unpriced_calls={unpriced} (the total is a floor)"
    logger.info(line)
    comment = f"models {spent.llm_usd:.4f} + DataForSEO {spent.dataforseo_usd:.4f}"
    if breakdown:
        comment += f"; {breakdown}"
    if unpriced:
        comment += f"; {unpriced} unpriced calls, so a floor"
    observability.record(ANALYSIS_COST_USD, round(spent.total_usd, 6), comment=comment)
