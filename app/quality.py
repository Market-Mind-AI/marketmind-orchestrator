"""What the run is judged on, beyond whether it finished.

Latency, token spend and cost come free from the trace. These are the checks
that need a definition before they can be a number, and the definition is the
work: the figures the narrative states that the evidence does not hold
(unsupported_numbers), the phrases the method forbids (language_violations),
and the tool calls that were not worth making (tool_call_faults). The
sentences without a figure are for a second model, and that one is the
evaluator Langfuse runs on every `explanation` generation, not this service.

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

