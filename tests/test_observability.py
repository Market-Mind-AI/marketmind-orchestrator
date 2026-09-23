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


# --- os fatos: o que foi baixado, e de que produto se trata -----------------
#
# 23/09/2026. O trace dizia quais nós rodaram e quanto custaram, e não dizia as
# duas coisas que separam uma execução cara de uma barata: se as fontes foram
# realmente consultadas ou se o banco já tinha o dado, e se o catálogo achou o
# produto ou criou outro. As duas são invisíveis no resultado — uma análise que
# baixa tudo e uma que não baixa nada têm exatamente a mesma forma.


def test_a_fact_is_written_to_the_log_even_with_no_tracing(monkeypatch, caplog) -> None:
    """Sem chaves não há trace, e o fato não pode sumir junto: o log é onde
    alguém lendo UMA execução com problema descobre o que houve."""
    import logging

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with caplog.at_level(logging.INFO, logger="app.observability"):
        observability.note("data_sources", summary="World Bank: 0 fetched, 12 reused from the database")

    assert "FACT data_sources" in caplog.text
    assert "12 reused" in caplog.text


def test_a_broken_tracer_never_breaks_the_run(monkeypatch, caplog) -> None:
    """Uma medição que quebra a coisa medida é pior que medição nenhuma — a
    mesma regra de `record`, que esta função precisa herdar e não herdaria de
    graça."""
    import logging
    from unittest.mock import patch

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    with patch("langfuse.get_client", side_effect=RuntimeError("langfuse fora do ar")):
        with caplog.at_level(logging.INFO, logger="app.observability"):
            observability.note("catalog_resolution", summary="'Picanha Swift': created a NEW catalogue entry")

    assert "FACT catalog_resolution" in caplog.text, "a linha do log sai mesmo com o trace quebrado"
