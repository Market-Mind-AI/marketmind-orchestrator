"""Where a run's state lives between steps, and across restarts.

The checkpointer writes a snapshot after each super-step, keyed by thread_id.
Re-invoking the graph with the same thread_id resumes from the last completed
node instead of starting over, which is the whole recovery strategy: every step
is idempotent, so resuming is re-running.

What goes in it is REFERENCE, not data. The identifiers, which countries are
done, which sources have been tried, the deadline. The indicator values, the
scores and the evidence packet stay in the scoring service. Putting them here
too would write megabytes per super-step and leave two versions of the truth
about one analysis, which is worse than slow.

One thing this is NOT: a scheduler. It records where a run stopped, but nothing
here notices that it stopped. Something outside has to re-invoke the graph with
that thread_id -- otherwise the state is durable and never read, and an analysis
that should insist for up to 24 hours simply stops.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from app.config import get_settings


class CheckpointerNotConfigured(RuntimeError):
    """Raised at startup rather than degrading quietly.

    Durability that is silently absent is worse than none: nothing looks wrong
    until a pod restarts mid-analysis and the run is gone, which is exactly the
    moment nobody is watching.
    """


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[Any]:
    """Yield a checkpointer for the lifetime of the application."""
    settings = get_settings()
    kind = settings.checkpointer.strip().lower()

    if kind == "memory":
        # Asked for by name, never fallen back to. Everything is lost on
        # restart, which is fine for a test and never fine anywhere else.
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return

    if kind != "postgres":
        raise CheckpointerNotConfigured(f"Unknown CHECKPOINTER '{settings.checkpointer}'; use 'postgres' or 'memory'.")

    if not settings.checkpointer_database_url:
        raise CheckpointerNotConfigured(
            "CHECKPOINTER is 'postgres' but CHECKPOINTER_DATABASE_URL is not set. "
            "Set it, or ask for CHECKPOINTER=memory explicitly and accept losing runs on restart."
        )

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(settings.checkpointer_database_url) as saver:
        # Creates its own tables if they are not there. Safe to repeat, so it
        # runs on every start rather than being a deployment step someone has
        # to remember.
        #
        # They land in whichever database the URL points at, which today is the
        # one every service shares. Nothing here assumes that: the day each
        # service gets its own database back, this URL changes and nothing else
        # does.
        await saver.setup()
        yield saver
