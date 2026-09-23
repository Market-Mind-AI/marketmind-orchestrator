import pytest

from app.checkpointer import CheckpointerNotConfigured, open_checkpointer, open_store
from app.config import get_settings


async def test_memory_has_to_be_asked_for_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKPOINTER", "memory")
    get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None

    async with open_checkpointer() as checkpointer:
        assert checkpointer.__class__.__name__ == "InMemorySaver"


# --- the store: the same switch, read as a pair with the checkpointer ---


async def test_the_store_follows_the_same_memory_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKPOINTER", "memory")

    async with open_store() as store:
        assert store.__class__.__name__ == "InMemoryStore"


async def test_the_store_refuses_postgres_without_a_connection_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpointer already fails at startup for this. The store shares its
    settings, so it must fail the same way rather than compiling a graph with
    durable runs and silently no memory."""
    monkeypatch.setenv("CHECKPOINTER", "postgres")
    for name in ("CHECKPOINTER_DATABASE_URL", "CHECKPOINTER_DB_HOST", "CHECKPOINTER_DB_NAME", "CHECKPOINTER_DB_USER"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(CheckpointerNotConfigured):
        async with open_store():
            pass


async def test_the_store_rejects_the_same_unknown_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKPOINTER", "redis")

    with pytest.raises(CheckpointerNotConfigured):
        async with open_store():
            pass


async def test_postgres_without_a_connection_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never a silent downgrade to memory. Durability that quietly is not there
    is worse than none: nothing looks wrong until a pod restarts mid-analysis,
    which is exactly when nobody is watching."""
    monkeypatch.setenv("CHECKPOINTER", "postgres")
    for name in ("CHECKPOINTER_DATABASE_URL", "CHECKPOINTER_DB_HOST", "CHECKPOINTER_DB_NAME", "CHECKPOINTER_DB_USER"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(CheckpointerNotConfigured) as excinfo:
        async with open_checkpointer():
            pass

    assert "CHECKPOINTER_DB_HOST" in str(excinfo.value)


async def test_an_unknown_kind_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKPOINTER", "redis")

    with pytest.raises(CheckpointerNotConfigured):
        async with open_checkpointer():
            pass


async def test_the_same_thread_resumes_and_a_different_one_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the checkpointer buys, stated plainly.

    Re-invoking with the same thread_id continues the run that was already
    there. That is why the thread_id is the analysis id: a generated one would
    make every retry a second, separately billed run of the same analysis.
    """
    monkeypatch.setenv("CHECKPOINTER", "memory")

    from operator import add
    from typing import Annotated, TypedDict

    from langgraph.graph import END, StateGraph

    class Counting(TypedDict):
        steps: Annotated[list[str], add]

    def step(state: Counting) -> Counting:
        return {"steps": ["ran"]}

    async with open_checkpointer() as checkpointer:
        builder = StateGraph(Counting)
        builder.add_node("step", step)
        builder.set_entry_point("step")
        builder.add_edge("step", END)
        graph = builder.compile(checkpointer=checkpointer)

        first = await graph.ainvoke({"steps": []}, config={"configurable": {"thread_id": "analysis-1"}})
        again = await graph.ainvoke({"steps": []}, config={"configurable": {"thread_id": "analysis-1"}})
        other = await graph.ainvoke({"steps": []}, config={"configurable": {"thread_id": "analysis-2"}})

    assert first["steps"] == ["ran"]
    assert again["steps"] == ["ran", "ran"], "the same thread continues the run that was there"
    assert other["steps"] == ["ran"], "a different analysis starts clean"


async def test_the_url_is_built_from_the_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parts, not a URL, matching every other service here. A URL in the vault
    would carry a second copy of the database password, and changing the
    password would then mean remembering two places."""
    monkeypatch.setenv("CHECKPOINTER_DB_HOST", "pg.example")
    monkeypatch.setenv("CHECKPOINTER_DB_NAME", "marketmind")
    monkeypatch.setenv("CHECKPOINTER_DB_USER", "marketmind")
    monkeypatch.setenv("CHECKPOINTER_DB_PASSWORD", "p@ss word/1")
    monkeypatch.delenv("CHECKPOINTER_DATABASE_URL", raising=False)

    url = get_settings().resolved_checkpointer_url

    assert url == "postgresql://marketmind:p%40ss%20word%2F1@pg.example:5432/marketmind", (
        "a password with reserved characters has to survive being put in a URL"
    )


async def test_a_whole_url_still_wins_for_a_local_one_liner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKPOINTER_DATABASE_URL", "postgresql://u:p@localhost:5436/db")
    monkeypatch.setenv("CHECKPOINTER_DB_HOST", "ignored.example")

    assert get_settings().resolved_checkpointer_url == "postgresql://u:p@localhost:5436/db"


async def test_partial_connection_details_are_not_half_a_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host with no database name is a configuration someone started and did
    not finish, and it fails at startup like any other missing one."""
    monkeypatch.setenv("CHECKPOINTER", "postgres")
    monkeypatch.setenv("CHECKPOINTER_DB_HOST", "pg.example")
    monkeypatch.delenv("CHECKPOINTER_DB_NAME", raising=False)
    monkeypatch.delenv("CHECKPOINTER_DATABASE_URL", raising=False)

    with pytest.raises(CheckpointerNotConfigured):
        async with open_checkpointer():
            pass
