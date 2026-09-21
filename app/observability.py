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

from functools import lru_cache
from typing import Any

from app.config import get_settings


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
