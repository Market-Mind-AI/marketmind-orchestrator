"""Where a run's state lives between steps, and across restarts -- and, next to
it, where the agent's memory of runs before this one lives.

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

The store below is a different kind of memory: not one run's own progress, but
what the agent has learned across runs -- today, which markets the sources do
not hold data for. It is opened the same way, from the same connection, so the
two switch together and neither is a surprise once the other is understood.
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

    url = settings.resolved_checkpointer_url
    if not url:
        raise CheckpointerNotConfigured(
            "CHECKPOINTER is 'postgres' but no connection is configured. Set "
            "CHECKPOINTER_DB_HOST/NAME/USER/PASSWORD (or CHECKPOINTER_DATABASE_URL), "
            "or ask for CHECKPOINTER=memory explicitly and accept losing runs on restart."
        )

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(url) as saver:
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


@asynccontextmanager
async def open_store() -> AsyncIterator[Any]:
    """Yield the long-term memory store for the lifetime of the application.

    Same CHECKPOINTER_* switch and connection as the checkpointer above, on
    purpose: a store is meaningless without somewhere durable to put it, so it
    rides on the same decision rather than introducing a second one nobody
    asked for. "memory" here means the same thing it means there -- kept in
    the process, lost on restart, fine for a test and never fine anywhere else.

    Unlike the checkpointer, nothing downstream REQUIRES this to be configured:
    every node that reads or writes it guards the call, so an application that
    never opens a store -- or one whose store call fails -- runs exactly as it
    did before memory existed. Durability of a run's own progress is a hard
    requirement; remembering what past runs found out is not.
    """
    settings = get_settings()
    kind = settings.checkpointer.strip().lower()

    if kind == "memory":
        from langgraph.store.memory import InMemoryStore

        yield InMemoryStore()
        return

    if kind != "postgres":
        raise CheckpointerNotConfigured(f"Unknown CHECKPOINTER '{settings.checkpointer}'; use 'postgres' or 'memory'.")

    url = settings.resolved_checkpointer_url
    if not url:
        raise CheckpointerNotConfigured(
            "CHECKPOINTER is 'postgres' but no connection is configured. Set "
            "CHECKPOINTER_DB_HOST/NAME/USER/PASSWORD (or CHECKPOINTER_DATABASE_URL), "
            "or ask for CHECKPOINTER=memory explicitly and accept losing runs on restart."
        )

    from langgraph.store.postgres import AsyncPostgresStore

    async with AsyncPostgresStore.from_conn_string(url) as store:
        # Same reasoning as the saver's setup() above: creates its tables if
        # they are not there, safe to repeat, so it runs on every start.
        await store.setup()
        yield store
