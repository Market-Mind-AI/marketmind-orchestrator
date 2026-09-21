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
from contextlib import ExitStack, contextmanager
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
def run(analysis_id: str) -> Iterator[None]:
    """Hold one analysis's trace open for the length of its graph run.

    The measures are recorded from inside the nodes, and scoring "the current
    trace" needs a span that is current there. The LangChain handler opens spans
    but makes none of them current, so without this every score was dropped with
    a warning. The handler nests the graph under this span instead of starting a
    trace of its own.

    The trace id is derived from the analysis id: the run can be found in
    Langfuse from the record, and a resume by the same id lands in the same trace.
    """
    if _handler() is None:
        yield
        return

    from langfuse import Langfuse, get_client

    with ExitStack() as stack:
        try:
            stack.enter_context(
                get_client().start_as_current_observation(
                    name="analysis",
                    trace_context={"trace_id": Langfuse.create_trace_id(seed=analysis_id)},
                    metadata={"analysis_id": analysis_id},
                )
            )
        except Exception:  # noqa: BLE001 — tracing never fails an analysis
            logger.debug("Could not open the trace for analysis %s", analysis_id, exc_info=True)
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
