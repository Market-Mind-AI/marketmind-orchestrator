"""What a model call costs, in dollars, by the price table Langfuse holds.

The scoring service prices the calls it makes the same way (its app/pricing.py)
and reports the dollars on its response headers. This copy exists for the one
model call this service makes itself — the hallucination judge — so the run's
total has no hole in it. Same table, same rule: a project's own definitions
win over the ones Langfuse manages, and a call that cannot be priced is None,
never zero.

Reading the table never raises and never blocks a request: it runs from a
thread at startup and is retried until it succeeds (see main.py).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app import observability

logger = logging.getLogger(__name__)

_PAGE = 100


@dataclass(frozen=True)
class Definition:
    name: str
    pattern: re.Pattern[str]
    prices: Mapping[str, float]


_table: list[Definition] = []


def load() -> int:
    """Read every model definition Langfuse holds. Returns how many are
    usable afterwards; keeps the previous table when the read fails."""
    if not observability.tracing_enabled():
        return len(_table)
    try:
        from langfuse import get_client

        client = get_client()
        found: list[Any] = []
        page = 1
        while True:
            answer = client.api.models.list(limit=_PAGE, page=page)
            found.extend(answer.data)
            if page >= (getattr(answer.meta, "total_pages", None) or 1):
                break
            page += 1
    except Exception:  # noqa: BLE001 — pricing never fails the service
        logger.warning("Could not read the model prices from Langfuse; the judge's cost stays unknown", exc_info=True)
        return len(_table)
    _table[:] = _definitions(found)
    logger.info("Loaded %d model price definitions from Langfuse", len(_table))
    return len(_table)


def _definitions(models: list[Any]) -> list[Definition]:
    own: list[Definition] = []
    managed: list[Definition] = []
    for model in models:
        try:
            pattern = re.compile(model.match_pattern)
        except (re.error, TypeError):
            continue
        prices = {key: float(value.price) for key, value in (model.prices or {}).items() if value.price is not None}
        definition = Definition(model.model_name, pattern, prices)
        (managed if getattr(model, "is_langfuse_managed", False) else own).append(definition)
    return own + managed


def loaded() -> bool:
    return bool(_table)


def price(model: str, usage: Mapping[str, int] | None) -> float | None:
    """USD for one call, or None when it cannot be priced."""
    if not usage or not _table or not model:
        return None
    definition = next((d for d in _table if d.pattern.search(model)), None)
    if definition is None:
        return None
    total = 0.0
    for key, count in usage.items():
        if key == "total" or not count:
            continue
        unit = definition.prices.get(key)
        if unit is None:
            return None
        total += count * unit
    return total
