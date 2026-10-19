"""What the run is judged on, beyond whether it finished.

Latency and token spend come free from the trace. These are the three that
need a definition before they can be a number, and the definition is the work.

The hallucination check is only possible because no number in the answer comes
from a model: the narrative is written from a deterministic evidence packet, so
a figure that is not in the packet is not supported by anything. If the MOS ever
came out of a model, this file could not exist.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Phrases the narrative must not use, and why. These are the language rules the
# method already commits to; they are rules, not judgement, so they are checked
# here rather than sent to a model.
_FORBIDDEN = [
    (
        re.compile(r"\b(proven|confirmed|actual|real)\s+demand\b", re.I),
        "search interest is not proven demand — the sources measure people searching, not people buying",
    ),
    (
        re.compile(r"\bguarantee(s|d)?\b", re.I),
        "the score is an opportunity reading, never a guarantee",
    ),
]


def numbers_in(text: str) -> list[float]:
    """Every figure the narrative states.

    Thousands separators and decimal commas both appear, and a number written
    either way has to be recognised or the check quietly passes everything.
    """
    out: list[float] = []
    for raw in re.findall(r"-?\d[\d.,]*", text or ""):
        cleaned = raw.rstrip(".,")
        # 1.234,56 (European) vs 1,234.56 (English): the LAST separator is the
        # decimal one, the rest group thousands.
        if "," in cleaned and "." in cleaned:
            decimal = max(cleaned.rfind(","), cleaned.rfind("."))
            cleaned = re.sub(r"[.,]", "", cleaned[:decimal]) + "." + cleaned[decimal + 1 :]
        else:
            # One kind of separator, so it is either the decimal point or
            # thousands grouping. Groups of exactly three, more than once, or a
            # single group of three after a leading part, mean grouping.
            #
            # Getting this wrong is not a rounding error: 1.500.000 read as
            # nothing at all means a fabricated million-search claim passes the
            # check silently, which is the exact failure this file exists to
            # catch.
            for sep in (",", "."):
                if sep not in cleaned:
                    continue
                parts = cleaned.split(sep)
                grouped = len(parts) > 2 and all(len(p) == 3 for p in parts[1:])
                grouped = grouped or (len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3)
                cleaned = cleaned.replace(sep, "") if grouped else cleaned.replace(sep, ".")
        try:
            out.append(float(cleaned))
        except ValueError:
            # A shape nobody anticipated. Said out loud rather than dropped: a
            # checker that silently ignores what it cannot read approves
            # everything it cannot read.
            logger.warning("Could not read %r as a number while checking the narrative", raw)
    return out


def numbers_in_packet(packet: dict[str, Any]) -> set[float]:
    """Every figure the evidence supports, at any depth."""
    found: set[float] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            found.add(float(node))
        elif isinstance(node, str):
            for value in numbers_in(node):
                found.add(value)

    walk(packet)
    return found


def unsupported_numbers(narrative: str, packet: dict[str, Any], *, tolerance: float = 0.5) -> list[float]:
    """Figures the narrative states that the evidence does not contain.

    Rounding is allowed — a packet holding 40.0219 supports "40" — because
    rounding is how the text is meant to read. Anything further apart is a
    figure nobody computed.

    Years and small counts are skipped: "the last 3 years" and "2026" are
    prose, not claims about this product's data, and treating them as
    unsupported would bury the real finding under noise.
    """
    supported = numbers_in_packet(packet)
    if not supported:
        return []

    unsupported: list[float] = []
    for value in numbers_in(narrative):
        if abs(value) <= 12 or 1900 <= value <= 2100:
            continue
        if not _is_supported(value, supported, tolerance):
            unsupported.append(value)
    return unsupported


# The same number written at a different magnitude: a percentage, or thousands,
# or "212.8 million" for a population the packet holds as 212,812,405. All of
# these are the packet's own figure restated, and counting them as unsupported
# buries the real finding under ordinary writing.
_MAGNITUDES = (1, 1e2, 1e3, 1e6, 1e9)

# Rounding to millions cannot be checked with an absolute tolerance. "212.8
# million" and 212,812,405 are twelve thousand apart, which is nothing at that
# scale and everything at the scale of a score. Proportion is the right measure
# once the magnitudes differ — a fabricated figure landing within a tenth of a
# percent of a real one, at an exact power of ten, is not a way anyone gets a
# number wrong.
_RELATIVE_TOLERANCE = 1e-3


def _is_supported(value: float, supported: set[float], tolerance: float) -> bool:
    """Whether the evidence holds this figure, at any ordinary magnitude.

    Rounding is allowed in both directions — a packet holding 40.0219 supports
    "40", and a packet holding 212,812,405 supports "212.8 million".
    """
    for ok in supported:
        for scale in _MAGNITUDES:
            for stated, held in ((value * scale, ok), (value, ok * scale)):
                if abs(stated - held) <= tolerance:
                    return True
                if held and abs(stated - held) <= abs(held) * _RELATIVE_TOLERANCE:
                    return True
    return False


def language_violations(narrative: str) -> list[str]:
    """Rules the narrative broke. Rules, not judgement — so no model is asked."""
    return [reason for pattern, reason in _FORBIDDEN if pattern.search(narrative or "")]


def tool_call_faults(coverage_tasks: list[dict[str, Any]], countries: list[str]) -> dict[str, int]:
    """Two exact, free measures of how well the run used its tools.

    "Accuracy" alone is not measurable. Split, it is: the SAME collection
    dispatched twice in one run is redundant — that is where cost leaks — and a
    country nobody asked about is a call that should not have happened at all.

    What counts as "the same collection" is the whole of it: the country, the
    indicator and the period. Keying on the country alone was wrong and wrong in
    one direction — every country legitimately gets two dispatches, one for
    search volume and one for the related keywords, and counting the second as a
    repeat of the first reported a third of every analysis as waste. An analysis
    of three markets came back at 0.5 with nothing whatsoever wrong with it.
    """
    wanted = set(countries)
    seen: dict[tuple[str, str, str], int] = {}
    stray = 0
    for task in coverage_tasks or []:
        country = task.get("country")
        if country is None:
            continue
        if country not in wanted:
            stray += 1
        key = (country, str(task.get("indicator_code") or ""), str(task.get("period_key") or ""))
        seen[key] = seen.get(key, 0) + 1
    return {
        "redundant": sum(count - 1 for count in seen.values() if count > 1),
        "off_request": stray,
    }


# --- The judge: for the sentences that carry no number ---


class JudgeNotConfigured(RuntimeError):
    """No judge model is configured, so nothing was judged."""


async def judge_support(narrative: str, packet: dict[str, Any]) -> dict[str, Any] | None:
    """Ask a second model whether the packet supports what the narrative claims.

    Only for the part the deterministic check cannot reach: sentences that make
    a claim without stating a figure. The numbers are already checked exactly,
    and a model is a worse instrument for something that can be compared.

    Returns None when no judge is configured. Not an error and not a failure —
    the analysis is finished and correct either way, and a missing judge means
    one signal is absent, not that anything is wrong. It is said in the log so
    an empty score is never mistaken for a clean one.
    """
    from app.config import get_settings

    settings = get_settings()
    if not settings.judge_api_key:
        logger.info(
            "Hallucination judge is not configured (JUDGE_API_KEY unset), so the narrative was not judged. "
            "The deterministic checks — every figure against the evidence, and the language rules — still ran."
        )
        return None

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.judge_api_key)
    prompt = (
        "You are checking one business narrative against the evidence it was written from.\n\n"
        "Answer ONLY about claims the evidence does not support. A claim is unsupported when the "
        "evidence contains nothing that would establish it — not when it is merely phrased "
        "differently, rounded, or summarised.\n\n"
        f"EVIDENCE:\n{packet}\n\nNARRATIVE:\n{narrative}"
    )
    schema = {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["supported", "unsupported_claims"],
        "additionalProperties": False,
    }

    # The one model call this service makes itself, traced like the ones the
    # scoring service makes (its TracedProvider): a generation in the run's
    # trace with model, tokens and level, so the run's cost includes the
    # measuring and a judge that fails shows where it failed.
    generation = _open_judge_generation(settings.judge_model, prompt)
    try:
        response = await client.chat.completions.create(
            model=settings.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "support", "schema": schema, "strict": True},
            },
        )
    except Exception as exc:  # noqa: BLE001 — a judge that fails must never fail the analysis
        logger.warning("The hallucination judge could not be reached, so the narrative was not judged: %s", exc)
        _close_judge_generation(generation, level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
        return None

    # The judge is a model call the run made, so it belongs in the run's token
    # figure. Measuring a thing is not free, and a cost report that leaves out
    # the measuring understates what the run cost.
    from app import spend

    judged = getattr(response, "usage", None)
    prompt_tokens = getattr(judged, "prompt_tokens", None)
    completion_tokens = getattr(judged, "completion_tokens", None)
    spend.add_call(prompt_tokens, completion_tokens)

    import json

    content = response.choices[0].message.content or "{}"
    if prompt_tokens is None and completion_tokens is None:
        _close_judge_generation(
            generation,
            output=content,
            level="WARNING",
            status_message="the provider reported no token usage; the call is not priced",
        )
    else:
        _close_judge_generation(
            generation,
            output=content,
            usage_details={"input": prompt_tokens or 0, "output": completion_tokens or 0},
        )
    return json.loads(content)


def _open_judge_generation(model: str, prompt: str) -> Any | None:
    """The judge's generation in the run's trace, or None when untraced.
    Never raises: tracing the judge must not fail the judge."""
    from app import observability

    if not observability.tracing_enabled():
        return None
    try:
        from langfuse import get_client

        return get_client().start_observation(
            name="narrative_judge",
            as_type="generation",
            model=model,
            input={"user": prompt},
            metadata={"provider": "AsyncOpenAI", "site": "narrative_judge"},
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not open the judge's generation", exc_info=True)
        return None


def _close_judge_generation(generation: Any | None, **update: Any) -> None:
    if generation is None:
        return
    try:
        generation.update(**{k: v for k, v in update.items() if v is not None})
        generation.end()
    except Exception:  # noqa: BLE001
        logger.debug("Could not close the judge's generation", exc_info=True)
