"""What a run spends on models, added up across the calls it makes.

The agent decides; the scoring service makes the model calls those decisions
cause and reports what each one cost on its response headers (see that
service's report_llm_usage middleware). This adds them up for the run, plus the
judge, which is the one model this service calls itself.

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

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Mapping


@dataclass
class Spend:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    # Model calls whose provider reported no figures. Kept apart so a total of
    # zero over several calls reads as "not reported" rather than as "free".
    calls_without_usage: int = 0
    _seen: set[str] = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __str__(self) -> str:
        parts = [f"{self.input_tokens} in + {self.output_tokens} out over {self.calls} model calls"]
        if self.calls_without_usage:
            parts.append(f"{self.calls_without_usage} of them reported no usage")
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
    """Add one scoring-service response's reported model spend to the run."""
    if "X-LLM-Calls" not in headers:
        return
    spend = _current.get()
    if spend is None:
        return
    spend.input_tokens += _as_int(headers.get("X-LLM-Input-Tokens"))
    spend.output_tokens += _as_int(headers.get("X-LLM-Output-Tokens"))
    spend.calls += _as_int(headers.get("X-LLM-Calls"))
    spend.calls_without_usage += _as_int(headers.get("X-LLM-Calls-Without-Usage"))


def add_call(input_tokens: int | None, output_tokens: int | None) -> None:
    """Add one model call this service made itself. Today that is the judge."""
    spend = _current.get()
    if spend is None:
        return
    spend.calls += 1
    if input_tokens is None and output_tokens is None:
        spend.calls_without_usage += 1
        return
    spend.input_tokens += input_tokens or 0
    spend.output_tokens += output_tokens or 0


def _as_int(value: str | None) -> int:
    try:
        return int(value) if value is not None else 0
    except ValueError:
        return 0
