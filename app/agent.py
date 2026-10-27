"""The decisions a fixed rule cannot make well, made by a model — inside a fence.

The graph's branches used to be the only judgement in a run, and they are
blunt: a country whose collection failed after the scoring service's own
retries was insisted on until three attempts or the deadline ran out, then the
whole analysis failed; a scoring service that timed out while measuring failed
the run on the spot. Whether to try again now, wait for a source to come back,
or deliver what is already scored with a note depends on things an `if` does not
weigh well together — which source, what it said, how much of the deadline is
left, how much of the request is already answered.

So at those points the model chooses. What it may choose is fenced three ways:

- the ACTIONS are a closed enum, and only the ones that make sense right now
  are offered (no retrying past the deadline, no delivering with nothing
  scored, recomputing once at most) — the fence is computed here, not asked of
  the model;
- every action is carried out by the same deterministic code paths the graph
  already had — going back to collect, waiting, crowning, delivering, giving
  up. The model never produces a number, a row or a sentence the buyer reads as
  a result; it produces a choice and the reason for it;
- the reason is recorded — the log, the trace, and the analysis progress line —
  so "why did it do that" has an answer in words.

When the model is not configured, not installed, or fails, the choice falls
back to the rule the graph applied before this existed. A run never waits on a
model to decide whether it may continue.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app import observability, spend
from app.config import get_settings

logger = logging.getLogger(__name__)


class Situation(StrEnum):
    # A country's collection failed after the scoring service's retries.
    COLLECTION_FAILED = "collection_failed"
    # A source — or the scoring service itself — is down or timed out.
    SOURCE_DOWN = "source_down"
    # The scored result does not hold together before delivery.
    INCOHERENT = "incoherent"


class Action(StrEnum):
    # Go back to collect now, within the deadline.
    RETRY_LATER = "retry_later"
    # Wait a while for the source to come back, then collect again.
    WAIT_AND_RESUME = "wait_and_resume"
    # Deliver the markets that scored; the others go out unscored, as the
    # scoring service left them.
    CONTINUE_WITHOUT = "continue_without"
    # Deliver what is scored, with a note saying what is missing and why.
    DELIVER_WITH_NOTE = "deliver_with_note"
    # Ask the scoring service to measure, score, crown and rebuild once more.
    RECOMPUTE_ONCE = "recompute_once"
    # Deliver as it is.
    DELIVER = "deliver"
    # End the run with a reason.
    GIVE_UP = "give_up"


# What each action does, in the words the model is given. Kept beside the enum
# so the description and the code path it names cannot drift apart.
ACTION_MEANING: dict[Action, str] = {
    Action.RETRY_LATER: "collect the missing countries again now (the scoring service already retried each source)",
    Action.WAIT_AND_RESUME: "wait for the sources to come back, then collect again — still within the deadline",
    Action.CONTINUE_WITHOUT: "deliver the scored countries now; the missing ones go out unscored",
    Action.DELIVER_WITH_NOTE: "deliver the scored countries now, with a note naming what is missing and why",
    Action.RECOMPUTE_ONCE: "ask the scoring service to re-measure, re-score and rebuild the evidence once",
    Action.DELIVER: "deliver the result as it is",
    Action.GIVE_UP: "fail the analysis with a reason; nothing is delivered",
}


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    # "model", "fallback" (the rule, because the model could not answer) or
    # "only_option" (nothing to choose between).
    decided_by: str

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action.value, "reason": self.reason, "decided_by": self.decided_by}


def fallback(situation: Situation, allowed: list[Action]) -> Action:
    """The rule the graph applied before a model chose. For a failed or
    unreachable source: insist while attempts and the deadline allow, then give
    up. For an incoherent result: deliver, which is what happened when nothing
    looked."""
    if situation is Situation.INCOHERENT:
        return Action.DELIVER if Action.DELIVER in allowed else allowed[0]
    if Action.RETRY_LATER in allowed:
        return Action.RETRY_LATER
    return Action.GIVE_UP if Action.GIVE_UP in allowed else allowed[0]


# A model that answers (action, reason) for the facts and the allowed actions.
Chooser = Callable[[Situation, dict[str, Any], list[Action]], Awaitable[tuple[str, str]]]


async def decide(situation: Situation, facts: dict[str, Any], allowed: list[Action]) -> Decision:
    """Choose one of `allowed`. Never raises, never returns anything else."""
    if not allowed:
        raise ValueError("a decision needs at least one allowed action")
    if len(allowed) == 1:
        return Decision(allowed[0], "the only action the situation allows", "only_option")

    chooser = _chooser()
    if chooser is None:
        action = fallback(situation, allowed)
        return Decision(action, "no decision model is configured; the fixed rule applies", "fallback")
    try:
        raw_action, reason = await chooser(situation, facts, allowed)
        action = Action(raw_action)
        if action not in allowed:
            raise ValueError(f"{raw_action!r} is not allowed here")
        return Decision(action, (reason or "").strip()[:500] or "no reason given", "model")
    except Exception as exc:  # noqa: BLE001 — the model may fail in any way; the rule does not
        logger.warning("The decision model could not decide (%s); the fixed rule applies", exc)
        action = fallback(situation, allowed)
        return Decision(action, f"the decision model failed ({type(exc).__name__}); the fixed rule applies", "fallback")


def for_progress(decision: Decision) -> str:
    """The line the buyer's screen shows. The model's reason goes in only when
    it states no figure: every number a buyer reads comes from the scoring
    service, and a reason is the model's words."""
    reason = decision.reason
    if decision.decided_by == "model" and re.search(r"\d", reason):
        reason = "reason recorded in the trace"
    return f"agent decision: {decision.action.value.replace('_', ' ')} — {reason}"


# --- the model --------------------------------------------------------------

_SYSTEM = """You decide what an automated market analysis does next when something went wrong.
You are given the situation, the facts, and the actions allowed right now.
Pick exactly one allowed action and give a one-sentence reason a buyer could read.
Weigh: whether the failure looks temporary (timeouts, 5xx, rate limits) or permanent,
how much time is left before the deadline, and how much of the request is already scored.
Never state numbers in the reason. Never invent data: you only choose what happens next."""


def _chooser() -> Chooser | None:
    """The model, when a key is configured; None otherwise, which means the
    fixed rule."""
    return _ask_claude if get_settings().anthropic_api_key else None


async def _ask_claude(situation: Situation, facts: dict[str, Any], allowed: list[Action]) -> tuple[str, str]:
    import anthropic

    settings = get_settings()
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": [a.value for a in allowed]},
            "reason": {"type": "string"},
        },
        "required": ["action", "reason"],
        "additionalProperties": False,
    }
    prompt = json.dumps(
        {
            "situation": situation.value,
            "facts": facts,
            "allowed_actions": {a.value: ACTION_MEANING[a] for a in allowed},
        },
        ensure_ascii=False,
        default=str,
    )
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=60.0)
    with observability.generation(
        "agent_decision", model=settings.decision_model, input={"situation": situation.value, "facts": facts}
    ) as generation:
        # Structured output, not a forced tool: the answer is one of the enum's
        # values by construction, and checked again by the caller.
        response = await client.messages.create(
            model=settings.decision_model,
            max_tokens=1024,
            system=_SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        spend.add_model_call(response.usage.input_tokens, response.usage.output_tokens)
        text = next(block.text for block in response.content if block.type == "text")
        answer = json.loads(text)
        if generation is not None:
            generation.update(
                output=answer,
                usage_details={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
            )
    return str(answer["action"]), str(answer["reason"])
