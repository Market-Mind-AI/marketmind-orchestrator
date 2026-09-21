import pytest

from app.checkpointer import CheckpointerNotConfigured, open_checkpointer
from app.config import get_settings


async def test_memory_has_to_be_asked_for_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKPOINTER", "memory")
    get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None

    async with open_checkpointer() as checkpointer:
        assert checkpointer.__class__.__name__ == "InMemorySaver"


async def test_postgres_without_a_url_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never a silent downgrade to memory. Durability that quietly is not there
    is worse than none: nothing looks wrong until a pod restarts mid-analysis,
    which is exactly when nobody is watching."""
    monkeypatch.setenv("CHECKPOINTER", "postgres")
    monkeypatch.delenv("CHECKPOINTER_DATABASE_URL", raising=False)

    with pytest.raises(CheckpointerNotConfigured) as excinfo:
        async with open_checkpointer():
            pass

    assert "CHECKPOINTER_DATABASE_URL" in str(excinfo.value)


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
