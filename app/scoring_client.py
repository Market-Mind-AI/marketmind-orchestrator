"""The scoring service's tools, as Python calls.

Everything the agent does to an analysis goes through here: registering it,
enriching its vocabulary, collecting what a country is missing, computing the
indicators and the MOS, building the evidence, narrating it, finishing it.

The agent decides WHAT to do. None of these decide anything: each one is a step
whose answer is the same every time it is asked, which is what lets a resumed
run re-run it safely.

Two failures are told apart deliberately, because they lead to opposite
decisions. A ScoringUnavailable is the service being unreachable or broken, and it
is worth retrying. A ScoringRejected is the service saying no -- a bad request, an
unknown entity, a missing token -- and retrying it just repeats the answer.
"""

from typing import Any

import httpx
from opentelemetry.propagate import inject

from app import spend
from app.config import get_settings

# The scoring service sits next to this one, and most calls answer in seconds.
# The ones that collect are the exception and get their own timeout (below).
_TIMEOUT = httpx.Timeout(600.0, connect=5.0)


def _collection_timeout(seconds: float | None) -> httpx.Timeout:
    """For the calls that collect: /coverage/ensure, /candidates/coverage and
    /champion/crown.

    The scoring service bounds ONE collection gap (dispatch, polling and the
    final poll — about 1952 s) and runs a country's gaps in sequence inside a
    single call, so these can take a multiple of that. Six hundred seconds cut
    them short while the scoring service was still collecting, and the run
    read its own impatience as a source being down. The caller passes what the
    run can still afford — its deadline — and outside a run the configured
    ceiling applies. Only the read is long; connecting stays quick, because a
    service that does not answer the door is down now, not busy.
    """
    if seconds is None:
        seconds = get_settings().scoring_collection_timeout_max_seconds
    return httpx.Timeout(seconds, connect=5.0)


class ScoringError(RuntimeError):
    """Base for anything that went wrong talking to the scoring service."""


class ScoringUnavailable(ScoringError):
    """Unreachable, timed out, or answered 5xx. Worth retrying."""


class ScoringRejected(ScoringError):
    """Answered 4xx: the request was wrong, not the moment. Retrying repeats it."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _why(exc: BaseException) -> str:
    """What went wrong, in words, even when the exception has none.

    httpx raises its timeouts with an EMPTY message, so a report built from
    `str(exc)` alone ended at the colon: "POST /internal/coverage/ensure
    failed: ". A World Bank call that hung for ten minutes reached the log as
    that blank line, and finding the cause meant reading another service's
    logs. When the exception says nothing, its class name is the fact.
    """
    return str(exc) or type(exc).__name__


class ScoringClient:
    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.scoring_api_url).rstrip("/")
        self._token = token if token is not None else settings.scoring_internal_token

    @property
    def base_url(self) -> str:
        return self._base_url

    def headers(self) -> dict[str, str]:
        """The auth header, for a caller that streams instead of using _call.

        Also carries the run's trace context (`traceparent`) when called during
        a run, so the model calls it causes land inside that trace too."""
        return self._headers()

    def _headers(self) -> dict[str, str]:
        # An absent token is not sent as an empty one: the service answers 401
        # either way, and a header that is present but empty reads like a
        # configuration that was attempted rather than one that was never made.
        headers = {"X-Internal-Token": self._token} if self._token else {}
        # The run's trace, so the model calls this request causes in the scoring
        # service land inside it. Outside a run nothing is current and nothing
        # is added.
        inject(headers)
        return headers

    async def _call(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=timeout or _TIMEOUT) as client:
                response = await client.request(method, url, json=json, headers=self._headers())
        except httpx.HTTPError as exc:
            raise ScoringUnavailable(f"{method} {path} failed: {_why(exc)}") from exc

        # Before the status check: a call that failed still burned the tokens it
        # burned before failing, and a run that retries three times has to
        # report all three.
        spend.add_response(response.headers)

        if 400 <= response.status_code < 500:
            raise ScoringRejected(response.status_code, _detail(response))
        if response.status_code >= 500:
            raise ScoringUnavailable(f"{method} {path} answered {response.status_code}: {_detail(response)}")
        return response.json() if response.content else {}

    # --- identity ---

    async def identify_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One photo (or one line of text) to a named product with a per-country
        vocabulary. The buyer sees and corrects this before anything is scored."""
        return await self._call("POST", "/catalog/identify-image", json=payload)

    async def countries(self) -> Any:
        """The markets that can be analysed, with the names and flags a report
        shows. Cached by the caller, not here."""
        return await self._call("GET", "/countries")

    async def read_report(self, request_id: str) -> dict[str, Any]:
        """The whole answer shaped for a reader — scores, narrative, evidence
        packet, brand/category. Distinct from read_analysis, which is the
        smaller shape a RESUMING RUN needs."""
        return await self._call("GET", f"/internal/analyses/{request_id}/report")

    # --- the analysis record ---

    async def register_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resolve the product and open the record, in one call.

        The same body as POST /analyze. Resolving here and registering in a
        second call was two halves of one decision split across services: the
        curated vocabulary and the buyer's edits reached the resolve, and the
        check of those edits and the brand fields were the registering side's
        to remember. The scoring service does both in one place now, and answers
        with what it resolved.
        """
        return await self._call("POST", "/internal/analyses/register", json=payload)

    async def read_analysis(self, request_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/internal/analyses/{request_id}")

    async def orphaned_analyses(self) -> list[str]:
        """Analyses that are running and that nobody is running.

        Asked once at startup. This process knows it just restarted; what it
        could not know is which analyses that restart orphaned — and without
        asking it had to wait for the scoring service's fifteen-minute sweep to
        notice, with the buyer watching a screen that says "in progress".
        """
        return [str(request_id) for request_id in await self._call("GET", "/internal/analyses/orphaned")]

    async def claim_analysis(self, request_id: str, *, runner: str) -> bool:
        """Whether this process may run it. False when another live process
        holds it — which is what lets two of these run side by side."""
        answer = await self._call("POST", f"/internal/analyses/{request_id}/claim?runner={runner}")
        return bool(answer.get("granted"))

    async def mark_running(self, request_id: str, *, deadline_at: str | None = None) -> dict[str, Any]:
        """`deadline_at` is how long this run may insist on a source that is
        down. The scoring service stores it the first time only, so a resumed
        run sending it again cannot push the end forward — and it is what that
        service reads to close an analysis left stuck past it."""
        body = {"deadline_at": deadline_at} if deadline_at else None
        return await self._call("POST", f"/internal/analyses/{request_id}/running", json=body)

    async def record_progress(self, request_id: str, message: str | None) -> dict[str, Any]:
        """Says what the run is waiting on, and that it is alive.

        Both, and they are read by different people. The sentence is for whoever
        is watching; the heartbeat is how a run that died with its pod is told
        apart from one that is legitimately taking hours.
        """
        return await self._call(
            "PATCH", f"/internal/analyses/{request_id}/progress", json={"message": message}
        )

    async def complete_analysis(
        self,
        request_id: str,
        *,
        result: list[dict[str, Any]],
        breakdown: dict[str, Any] | None = None,
        explanation: str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            "POST",
            f"/internal/analyses/{request_id}/complete",
            json={"result": result, "breakdown": breakdown, "explanation": explanation},
        )

    async def attach_narrative(self, request_id: str, narrative: str | None) -> dict[str, Any]:
        return await self._call(
            "POST", f"/internal/analyses/{request_id}/narrative", json={"narrative": narrative}
        )

    async def fail_analysis(self, request_id: str, error_message: str) -> dict[str, Any]:
        return await self._call(
            "POST", f"/internal/analyses/{request_id}/fail", json={"error_message": error_message}
        )

    # --- the work ---

    async def enrich_terms(
        self,
        entity_key: str,
        countries: list[str],
        *,
        with_candidates: bool = True,
        brand_key: str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            "POST",
            "/internal/enrich-terms",
            json={
                "entity_key": entity_key,
                "countries": countries,
                "with_candidates": with_candidates,
                "brand_key": brand_key,
            },
        )

    async def ensure_coverage(
        self,
        country_iso3: str,
        entity_key: str | None,
        period_key: str,
        brand_key: str | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Collect what this country is missing, and report what was tried.

        The tasks are the answer, not the data. A FAILED task is a source that
        is down and worth insisting on; a task that finished without failing and
        still left the indicator missing is a source that does not have that
        datum. Insisting on the second never ends.

        The vocabulary the country was searched for comes back with them. It is
        the one part of the pipeline a model wrote, and nothing downstream
        reveals it: a wrong native term produces a perfectly formed analysis of
        the wrong words.
        """
        return await self._call(
            "POST",
            "/internal/coverage/ensure",
            json={
                "country_iso3": country_iso3,
                "entity_key": entity_key,
                # The run's period, fixed when it started: the collection
                # tasks are stamped with it, and a clock read over there would
                # stamp two months on an analysis that crosses midnight.
                "period_key": period_key,
                "brand_key": brand_key,
            },
            timeout=_collection_timeout(timeout),
        )

    async def candidates_coverage(
        self,
        entity_key: str,
        requested_countries: list[str],
        period_key: str,
        brand_key: str | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Collect the reference markets, unless the period's champion is known.

        Cached means the collection was skipped: candidate demand is the slowest
        work in the system and re-running it produces the same answer. Only the
        flag is read here; who is champion is /champion/crown's answer.
        """
        return await self._call(
            "POST",
            "/internal/candidates/coverage",
            json={
                "entity_key": entity_key,
                "requested_countries": requested_countries,
                "period_key": period_key,
                "brand_key": brand_key,
            },
            timeout=_collection_timeout(timeout),
        )

    async def demand_absence(
        self, entity_key: str, countries: list[str], period_key: str, brand_key: str | None = None
    ) -> dict[str, Any]:
        """Which of these markets had no demand for this product (and brand),
        recently — read from the stored data, so it heals itself the moment a
        source gains volume, inside the demand's own freshness window. A
        market with a term still to collect is not absent: that is not knowing.
        Free: nothing is collected."""
        return await self._call(
            "POST",
            "/internal/markets/demand-absence",
            json={
                "entity_key": entity_key,
                "countries": countries,
                "period_key": period_key,
                "brand_key": brand_key,
            },
        )

    async def demand_ceiling(
        self, entity_key: str, countries: list[str], period_key: str, brand_key: str | None = None
    ) -> tuple[float | None, dict[str, Any] | None]:
        body = await self._call(
            "POST",
            "/internal/demand-ceiling",
            json={
                "entity_key": entity_key,
                "countries": countries,
                "period_key": period_key,
                "brand_key": brand_key,
            },
        )
        return body.get("demand_ceiling"), body.get("demand_ruler")

    async def crown_champion(
        self,
        entity_key: str,
        period_key: str,
        requested_countries: list[str],
        requested_results: list[dict[str, Any]],
        brand_key: str | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """The product's best market for the period, requested or not — decided
        and stored by the scoring service, which holds the ruler it has to be
        measured against.

        Called after /mos/compute and before /evidence, which reads what this
        stored. `champion_row` is the champion's MOS row when nobody asked for
        it, ready to be delivered next to the requested ones. It may collect
        macro data for a few candidates, so it gets the collecting timeout.
        """
        return await self._call(
            "POST",
            "/internal/champion/crown",
            json={
                "entity_key": entity_key,
                "period_key": period_key,
                "requested_countries": requested_countries,
                "requested_results": requested_results,
                "brand_key": brand_key,
            },
            timeout=_collection_timeout(timeout),
        )

    async def compute_indicators(
        self,
        entity_key: str | None,
        country_iso3: str,
        period_key: str,
        demand_ceiling: float | None,
        brand_key: str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            "POST",
            "/indicators/compute",
            json={
                "entity_key": entity_key,
                "country": country_iso3,
                "period_key": period_key,
                "brand_key": brand_key,
                "demand_ceiling": demand_ceiling,
            },
        )

    async def calibrate_weights(self, period_key: str) -> dict[str, float]:
        body = await self._call("POST", "/internal/weights/calibrate", json={"period_key": period_key})
        return body.get("weights", {})

    async def compute_mos(
        self, entity_key: str, countries: list[str], period_key: str, brand_key: str | None = None
    ) -> list[dict[str, Any]]:
        body = await self._call(
            "POST",
            "/mos/compute",
            json={
                "entity_key": entity_key,
                "countries": countries,
                "period_key": period_key,
                "brand_key": brand_key,
            },
        )
        return body.get("results", body if isinstance(body, list) else [])

    async def build_evidence(
        self,
        entity_key: str,
        countries: list[str],
        period_key: str,
        demand_ruler: dict[str, Any] | None = None,
        brand_key: str | None = None,
        language: str | None = None,
    ) -> dict[str, Any] | None:
        """The packet comes back whole: which countries were requested, which
        is the crowned champion, the display order and every label in
        `language`. Nothing is added to it here."""
        body = await self._call(
            "POST",
            "/internal/evidence",
            json={
                "entity_key": entity_key,
                "countries": countries,
                "period_key": period_key,
                "demand_ruler": demand_ruler,
                "brand_key": brand_key,
                "language": language,
            },
        )
        return body.get("packet")

    async def narrate(
        self,
        packet: dict[str, Any],
        product_name: str,
        user_prompt: str | None = None,
        *,
        brand: str | None = None,
        language: str | None = None,
    ) -> str | None:
        """`language` is the one the analysis asked for (the buyer's interface
        language, read back from the record). None leaves it to the scoring
        service's configured default."""
        body = await self._call(
            "POST",
            "/internal/narrate",
            json={
                "packet": packet,
                "product_name": product_name,
                "brand": brand,
                "user_prompt": user_prompt,
                "language": language,
            },
        )
        return body.get("narrative")

    async def language_check(self, narrative: str, language: str | None = None) -> dict[str, Any]:
        """The phrases the method forbids, found in a narrative, by the rules
        written in the narrative's own language. The rules live next to the
        narrative's texts; a copy here was English-only and passed every
        Portuguese and Spanish narrative unread."""
        return await self._call(
            "POST",
            "/internal/narrative/language-check",
            json={"narrative": narrative, "language": language},
        )


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)[:500]
