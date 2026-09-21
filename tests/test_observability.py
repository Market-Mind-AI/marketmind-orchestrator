import pytest

from app import observability


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    observability._handler.cache_clear()
    yield
    observability._handler.cache_clear()


def test_without_keys_there_is_no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowed, unlike a missing checkpointer: a missing trace loses an
    explanation, it does not corrupt an answer."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert observability.callbacks() == []
    assert observability.tracing_enabled() is False


def test_one_key_alone_is_not_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert observability.callbacks() == []


def test_with_both_keys_a_handler_is_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")

    handlers = observability.callbacks()

    assert len(handlers) == 1
    assert observability.tracing_enabled() is True


def test_a_measure_recorded_during_a_run_lands_on_that_runs_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """The five measures are recorded from inside graph nodes, where the
    LangChain handler has opened spans but none of them is the current one. With
    nothing current, score_current_trace() logged "No active span in current
    context" and dropped every score: they reached the log and never Langfuse.

    The trace is the analysis id's, so the run can be found from the record and
    a resume by the same id lands in the same trace."""
    from langfuse import Langfuse

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    scored: list[dict] = []
    monkeypatch.setattr(Langfuse, "create_score", lambda self, **score: scored.append(score))

    analysis_id = "a8c5d38b-db60-413c-b8f3-086a0befb83a"
    with observability.run(analysis_id):
        observability.record("task_completion_rate", 0.0)

    assert [s["name"] for s in scored] == ["task_completion_rate"]
    assert scored[0]["trace_id"] == Langfuse.create_trace_id(seed=analysis_id)


def test_a_run_without_tracing_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with observability.run("a8c5d38b-db60-413c-b8f3-086a0befb83a"):
        observability.record("task_completion_rate", 1.0)


@pytest.fixture
def clean_logging():
    """setup_logging binds the stdout it sees and then refuses to add a second
    handler, which is right for a service with one process and one stdout — and
    means a test has to start from a clean root to capture anything."""
    import logging

    root = logging.getLogger()
    saved, saved_level = root.handlers[:], root.level
    root.handlers.clear()
    yield
    root.handlers[:] = saved
    root.setLevel(saved_level)


def test_the_applications_own_lines_actually_reach_the_log(clean_logging) -> None:
    """Nothing configured logging, and uvicorn configures only its own loggers,
    so every logger.info in this codebase went to a root logger with no handler
    and was dropped. What was lost is exactly what is worth reading: the search
    terms each country was queried for, the five measures, and the line saying
    the judge is not configured.

    It passed every test, because caplog attaches its own handler — the
    assertions saw lines production never wrote. This one reads the stream the
    handler was actually given.
    """
    import io
    import logging
    import sys
    from unittest.mock import patch

    from app.logging_setup import setup_logging

    stream = io.StringIO()
    with patch.object(sys, "stdout", stream):
        setup_logging()
        logging.getLogger("app.nodes").info("SEARCH TERMS BRA: product=['picanha']")

    written = stream.getvalue()
    assert "SEARCH TERMS BRA" in written
    assert "app.nodes" in written, "the module has to be named, or a line says nothing about where it came from"


def test_configuring_twice_does_not_double_every_line(clean_logging) -> None:
    import io
    import logging
    import sys
    from unittest.mock import patch

    from app.logging_setup import setup_logging

    stream = io.StringIO()
    with patch.object(sys, "stdout", stream):
        setup_logging()
        setup_logging()
        logging.getLogger("app.nodes").info("once")

    assert stream.getvalue().count("once") == 1


def test_the_tool_calls_do_not_bury_the_lines_worth_reading(clean_logging) -> None:
    """httpx logs one INFO line per request and a run makes hundreds of tool
    calls."""
    import logging

    from app.logging_setup import setup_logging

    setup_logging()

    assert logging.getLogger("httpx").level == logging.WARNING
