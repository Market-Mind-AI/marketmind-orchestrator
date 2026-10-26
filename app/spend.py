"""What a run spends, added up across the calls it makes: tokens on models,
and dollars at DataForSEO for the demand data.

The agent decides; the scoring service makes the model calls those decisions
cause and reports what each one cost on its response headers (see that
service's report_llm_usage middleware). This adds them up for the run. The
same headers carry
what the Google Ads collector was billed by DataForSEO for the collections the
run caused — the one cost of an analysis that is not a model.

A context variable holding a mutable tally, opened once per run. LangGraph runs
nodes in tasks derived from the request's context, and a child task inherits the
same object, so a node that calls a tool adds to the same tally as every other
node without any of them having to pass it along.

The tally lives in the process, not in the checkpoint. A run that resumes after
a restart therefore reports what it spent since resuming, and the tokens spent
before were reported by the run that spent them. Writing it into the state
instead would make every checkpoint carry a counter, and the two halves still
would not add up to one number anyone looks at.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class Spend:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    # Model calls whose provider reported no figures. Kept apart so a total of
    # zero over several calls reads as "not reported" rather than as "free".
    calls_without_usage: int = 0
    # What the model calls cost, in USD by the price table Langfuse holds
    # (the scoring service prices them and reports the dollars); by model, for
    # the breakdown; and how
    # many calls could not be priced, kept apart because unknown is not free.
    llm_usd: float = 0.0
    llm_calls_without_cost: int = 0
    models_usd: dict[str, float] = field(default_factory=dict)
    # What DataForSEO billed, in USD as it reports it, over the Google Ads
    # tasks this run caused; and how many of those carried no figure.
    dataforseo_usd: float = 0.0
    dataforseo_calls: int = 0
    dataforseo_calls_without_cost: int = 0
    _seen: set[str] = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_usd(self) -> float:
        """Models and DataForSEO together — everything the run paid for."""
        return self.llm_usd + self.dataforseo_usd

    def __str__(self) -> str:
        parts = [f"{self.input_tokens} in + {self.output_tokens} out over {self.calls} model calls"]
        if self.calls_without_usage:
            parts.append(f"{self.calls_without_usage} of them reported no usage")
        if self.dataforseo_calls:
            parts.append(f"${self.dataforseo_usd:.4f} at DataForSEO over {self.dataforseo_calls} tasks")
            if self.dataforseo_calls_without_cost:
                parts.append(f"{self.dataforseo_calls_without_cost} of them reported no cost")
        return "; ".join(parts)


_current: ContextVar[Spend | None] = ContextVar("run_spend", default=None)


@contextmanager
def tally() -> Iterator[Spend]:
    """Add up everything spent inside this block."""
    spend = Spend()
    token = _current.set(spend)
    try:
        yield spend
    finally:
        _current.reset(token)


def current() -> Spend:
    """The run's tally, or an empty one when no run is open (a tool called from
    a test or a script still works, it is simply not counted)."""
    return _current.get() or Spend()


def add_response(headers: Mapping[str, str]) -> None:
    """Add one scoring-service response's reported spend to the run: the
    model calls it made, and what the collector it called was billed."""
    spend = _current.get()
    if spend is None:
        return
    if "X-LLM-Calls" in headers:
        spend.input_tokens += _as_int(headers.get("X-LLM-Input-Tokens"))
        spend.output_tokens += _as_int(headers.get("X-LLM-Output-Tokens"))
        spend.calls += _as_int(headers.get("X-LLM-Calls"))
        spend.calls_without_usage += _as_int(headers.get("X-LLM-Calls-Without-Usage"))
        spend.llm_usd += _as_float(headers.get("X-LLM-Cost-USD"))
        spend.llm_calls_without_cost += _as_int(headers.get("X-LLM-Calls-Without-Cost"))
        for part in (headers.get("X-LLM-Cost-By-Model") or "").split(","):
            model, _, usd = part.partition("=")
            if model and usd:
                spend.models_usd[model] = spend.models_usd.get(model, 0.0) + _as_float(usd)
    if "X-DataForSEO-Calls" in headers:
        spend.dataforseo_usd += _as_float(headers.get("X-DataForSEO-Cost-USD"))
        spend.dataforseo_calls += _as_int(headers.get("X-DataForSEO-Calls"))
        spend.dataforseo_calls_without_cost += _as_int(headers.get("X-DataForSEO-Calls-Without-Cost"))


def _as_int(value: str | None) -> int:
    try:
        return int(value) if value is not None else 0
    except ValueError:
        return 0


def _as_float(value: str | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except ValueError:
        return 0.0
