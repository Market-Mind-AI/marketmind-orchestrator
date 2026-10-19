"""The judge is priced by the table Langfuse holds, like every other call."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import observability, pricing


def _model(name: str, pattern: str, prices: dict[str, float], *, managed: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=name,
        match_pattern=pattern,
        prices={k: SimpleNamespace(price=v) for k, v in prices.items()},
        is_langfuse_managed=managed,
    )


GPT = _model("gpt-5.1", r"(?i)^(openai/)?(gpt-5.1)$", {"input": 1.25e-6, "output": 1e-5})


@pytest.fixture(autouse=True)
def _empty_table():
    pricing._table.clear()
    observability._handler.cache_clear()
    yield
    pricing._table.clear()
    observability._handler.cache_clear()


def test_the_table_is_read_from_langfuse_when_tracing_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    cliente = MagicMock()
    cliente.api.models.list.return_value = SimpleNamespace(data=[GPT], meta=SimpleNamespace(total_pages=1))

    with patch("langfuse.get_client", return_value=cliente):
        assert pricing.load() == 1

    assert pricing.price("gpt-5.1", {"input": 1000, "output": 100}) == pytest.approx(1.25e-3 + 1e-3)
    assert pricing.price("claude-opus-5", {"input": 1}) is None


def test_without_tracing_there_is_no_table_and_nothing_is_priced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert pricing.load() == 0
    assert pricing.price("gpt-5.1", {"input": 1}) is None


def test_a_failed_read_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    with patch("langfuse.get_client", side_effect=RuntimeError("langfuse fora do ar")):
        assert pricing.load() == 0


async def test_the_judge_is_charged_by_the_table(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import quality, spend
    from tests.test_quality import _PACKET, _FakeOpenAI

    monkeypatch.setenv("JUDGE_API_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeOpenAI)
    pricing._table[:] = pricing._definitions([GPT])

    with spend.tally() as spent:
        await quality.judge_support("anything", _PACKET)

    # the fake judge answers 800 in / 40 out
    assert spent.models_usd == {"gpt-5.1": pytest.approx(800 * 1.25e-6 + 40 * 1e-5)}
    assert spent.llm_calls_without_cost == 0
