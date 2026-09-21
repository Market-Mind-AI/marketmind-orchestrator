#!/usr/bin/env python3
"""Prove the agent produces the same answer as the loop it replaces.

This is the gate on step 8 of the migration. The whole point of writing the
agent as a faithful reimplementation is that it can be checked this way, and
the check only works once: after the loop is deleted there is nothing left to
compare against.

It runs the SAME product, countries and period twice -- once through the
scoring service's own /analyze, once through the agent -- and compares the
scored markets field by field.

The second run reads what the first one collected, so the numbers have to match
exactly. If they do not, the difference is in the code, not in the data.

    python scripts/parity_check.py "air fryer" ESP PRT BRA

Needs, in the environment:
    SCORING_API_URL            default http://localhost:8004
    SCORING_INTERNAL_TOKEN     the same secret the scoring service checks
    ORCHESTRATOR_URL           default http://localhost:8081
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import httpx

SCORING = os.environ.get("SCORING_API_URL", "http://localhost:8004").rstrip("/")
ORCHESTRATOR = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8081").rstrip("/")
TOKEN = os.environ.get("SCORING_INTERNAL_TOKEN", "")

# What has to match. Ids and timestamps differ between two runs by definition,
# and the narrative is a model's prose about identical numbers -- comparing it
# would fail on wording and say nothing about the method.
COMPARED = (
    "country",
    "mos",
    "mos_before_cap",
    "capped_by",
    "topsis_score",
    "mos_interpretation",
    "availability_status",
    "excluded_reason",
)

TIMEOUT = httpx.Timeout(1800.0, connect=5.0)


def _headers() -> dict[str, str]:
    return {"X-Internal-Token": TOKEN} if TOKEN else {}


async def _wait_for(client: httpx.AsyncClient, request_id: str) -> dict[str, Any]:
    """Poll until the analysis reaches a terminal state.

    Polling rather than the SSE stream: the stream closes on the terminal
    status, and the narrative is attached after that -- reading it here would
    race a write that does not affect the numbers.
    """
    while True:
        response = await client.get(f"{SCORING}/analyze/{request_id}")
        response.raise_for_status()
        body = response.json()
        if body["status"] in ("done", "failed"):
            return body
        await asyncio.sleep(3)


async def run_legacy(client: httpx.AsyncClient, product: str, countries: list[str]) -> dict[str, Any]:
    print(f"  legacy  -> POST {SCORING}/analyze")
    response = await client.post(
        f"{SCORING}/analyze", json={"product_name": product, "countries": countries}
    )
    response.raise_for_status()
    request_id = response.json()["request_id"]
    print(f"  legacy  -> analysis {request_id}")
    return await _wait_for(client, request_id)


async def run_agent(client: httpx.AsyncClient, product: str, countries: list[str]) -> dict[str, Any]:
    resolved = await client.post(
        f"{SCORING}/catalog/resolve", json={"text": product, "source": "parity-check"}
    )
    resolved.raise_for_status()
    entity = resolved.json()

    registered = await client.post(
        f"{SCORING}/internal/analyses",
        headers=_headers(),
        json={
            "entity_key": entity["entity_key"],
            "canonical_title": entity["canonical_title"],
            "countries": countries,
        },
    )
    registered.raise_for_status()
    request_id = registered.json()["request_id"]
    print(f"  agent   -> analysis {request_id}")

    print(f"  agent   -> POST {ORCHESTRATOR}/run-analysis")
    async with client.stream(
        "POST", f"{ORCHESTRATOR}/run-analysis", json={"analysis_id": request_id}
    ) as stream:
        stream.raise_for_status()
        async for line in stream.aiter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if "error" in event:
                    print(f"  agent   !! {event['error']}")
                else:
                    print(f"  agent   .. {', '.join(event.keys())}")

    return await _wait_for(client, request_id)


def compare(legacy: dict[str, Any], agent: dict[str, Any]) -> list[str]:
    """Every difference that matters, or an empty list."""
    problems: list[str] = []

    if legacy["status"] != agent["status"]:
        problems.append(f"status: legacy={legacy['status']} agent={agent['status']}")

    def by_country(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {row["country"]: row for row in (body.get("result") or [])}

    left, right = by_country(legacy), by_country(agent)

    # A market present in one answer and not the other is the worst kind of
    # difference: it changes the ranking rather than a figure in it.
    for country in sorted(set(left) - set(right)):
        problems.append(f"{country}: in the legacy answer, missing from the agent's")
    for country in sorted(set(right) - set(left)):
        problems.append(f"{country}: in the agent's answer, missing from the legacy one")

    for country in sorted(set(left) & set(right)):
        for field in COMPARED:
            a, b = left[country].get(field), right[country].get(field)
            if _differs(a, b):
                problems.append(f"{country}.{field}: legacy={a!r} agent={b!r}")

    # The dimension subscores under each MOS. Two identical MOS values built
    # from different subscores would be a coincidence, not agreement.
    for country in sorted(set(left) & set(right)):
        for name, a, b in _subscores(legacy, agent, country):
            if _differs(a, b):
                problems.append(f"{country}.dimensions.{name}: legacy={a!r} agent={b!r}")

    return problems


def _differs(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is not b
    try:
        return abs(float(a) - float(b)) > 1e-9
    except (TypeError, ValueError):
        return str(a) != str(b)


def _subscores(legacy: dict[str, Any], agent: dict[str, Any], country: str):  # noqa: ANN202
    def dims(body: dict[str, Any]) -> dict[str, Any]:
        for entry in ((body.get("breakdown") or {}).get("countries") or []):
            if entry.get("country") == country:
                return entry.get("dimensions") or {}
        return {}

    left, right = dims(legacy), dims(agent)
    for name in sorted(set(left) | set(right)):
        yield name, _score_of(left.get(name)), _score_of(right.get(name))


def _score_of(dimension: Any) -> Any:
    if isinstance(dimension, dict):
        return dimension.get("subscore", dimension.get("score"))
    return dimension


async def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    product, countries = sys.argv[1], [c.upper() for c in sys.argv[2:]]

    if not TOKEN:
        print("SCORING_INTERNAL_TOKEN is not set; every /internal call will be refused.")
        return 2

    print(f"Product : {product}")
    print(f"Markets : {', '.join(countries)}\n")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        legacy = await run_legacy(client, product, countries)
        print()
        agent = await run_agent(client, product, countries)

    print()
    problems = compare(legacy, agent)
    if not problems:
        scored = len([r for r in (legacy.get("result") or []) if r.get("mos") is not None])
        print(f"IDENTICAL across {scored} scored market(s). Step 8 passes.")
        return 0

    print(f"{len(problems)} difference(s) — step 8 does not pass:")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
