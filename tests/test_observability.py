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
