"""Tracing for the graph.

Every run becomes a tree: which nodes ran, which tools they called, how many
tokens each cost, how long each took. That is the record that answers "why did
it do that" once the thing deciding is a model rather than a branch in this
file -- and it is the only such record, because a model's reasoning is not in
the code to be read.

Unconfigured means untraced, and that is allowed: a missing trace loses an
explanation, it does not corrupt an answer. It is said once at startup rather
than silently, so nobody discovers it while trying to explain a bad run.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _handler() -> Any | None:
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    return CallbackHandler()


def callbacks() -> list[Any]:
    """The callbacks to hand a graph run. Empty when tracing is not configured."""
    handler = _handler()
    return [handler] if handler is not None else []


def tracing_enabled() -> bool:
    return _handler() is not None


@contextmanager
def run_span(name: str, *, seed: str | None = None) -> Iterator[None]:
    """Hold one span open for the whole run, so everything inside it attaches.

    `seed` makes the trace id a function of the analysis id instead of random,
    so a run's trace can be reached from the id the rest of the system already
    has — which is how its scores are read back, and how a support question
    ("what happened in analysis X?") gets answered without a search.

    Without this, `record` silently lost every score. Measured in production on
    23/09/2026: each QUALITY line was followed by langfuse's "No active span in
    current context... will be skipped", so the five measures — completion,
    latency, tokens, tool accuracy, hallucination — reached the log and never
    reached Langfuse. They had been written weeks earlier and looked fine,
    because the log line is emitted first and unconditionally.

    The LangChain callback handler does attach a context per observation, but
    it does so inside its own callback invocation; a node body is not
    guaranteed to run in that context. Opening one span around the whole
    invocation is what makes the guarantee, and it costs one observation.

    `create_event` failed differently and worse — it created its own ROOT trace
    instead of skipping, so the facts would have landed in Langfuse detached
    from the analysis that produced them. Present but unfindable.

    A no-op when tracing is not configured, like everything else here.
    """
    if _handler() is None:
        yield
        return
    try:
        from langfuse import Langfuse, get_client

        trace_context = {"trace_id": Langfuse.create_trace_id(seed=seed)} if seed else None
        with get_client().start_as_current_observation(as_type="span", name=name, trace_context=trace_context):
            yield
    except Exception:  # noqa: BLE001 — a span that cannot open must not stop the run
        logger.debug("Could not open the run span %s", name, exc_info=True)
        yield


def record(name: str, value: float | str | bool, *, comment: str | None = None) -> None:
    """Attach one measure to the run's trace, and say it in the log.

    Both, because they answer different questions. The trace is where a run is
    compared with the ones before it; the log is where someone reading a single
    failing run finds out what was wrong with it.

    Never raises. A measurement that breaks the thing it measures is worse than
    no measurement.
    """
    line = f"{name}={value}" + (f" ({comment})" if comment else "")
    logger.info("QUALITY %s", line)

    handler = _handler()
    if handler is None:
        return
    try:
        from langfuse import get_client

        get_client().score_current_trace(
            name=name,
            value=1.0 if value is True else 0.0 if value is False else value,
            data_type="BOOLEAN" if isinstance(value, bool) else None,
            comment=comment,
        )
    except Exception:  # noqa: BLE001 — never let a score break a run
        logger.debug("Could not attach the score %s to the trace", name, exc_info=True)


def note(name: str, *, summary: str, detail: dict[str, Any] | None = None) -> None:
    """Attach one FACT to the run's trace, and say it in the log.

    A sibling of `record`, for a different kind of thing. A score is a number
    to compare across runs; this is what happened in this one — which sources
    were actually read, which product the catalogue matched. Those do not
    reduce to a number without losing the part someone needs.

    It lands as an event under the node that is running, so it reads in place
    in the trace tree rather than in a list of scores off to the side.

    Never raises, for the same reason `record` does not: a measurement that
    breaks the thing it measures is worse than no measurement.
    """
    logger.info("FACT %s: %s", name, summary)

    handler = _handler()
    if handler is None:
        return
    try:
        from langfuse import get_client

        get_client().create_event(name=name, output=summary, metadata=detail or {})
    except Exception:  # noqa: BLE001 — never let an observation break a run
        logger.debug("Could not attach the event %s to the trace", name, exc_info=True)
