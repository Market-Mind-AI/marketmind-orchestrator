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

# The scoring service sits next to this one, but some of these calls trigger
# collection from an upstream API and legitimately take minutes.
_TIMEOUT = httpx.Timeout(600.0, connect=5.0)


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

    async def _call(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> Any:
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.request(method, url, json=json, headers=self._headers())
        except httpx.HTTPError as exc:
            raise ScoringUnavailable(f"{method} {path} failed: {exc}") from exc

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

    async def resolve_product(
        self,
        text: str,
        *,
        source: str = "orchestrator",
        brand: str | None = None,
        category: str | None = None,
        localized_markets: dict[str, Any] | None = None,
        reference_candidates: list[dict[str, Any]] | None = None,
        target_age_min: int | None = None,
        target_age_max: int | None = None,
    ) -> dict[str, Any]:
        """Text to a stable entity. The brand PARTITIONS identity rather than
        being concatenated into the text that decides similarity.

        `localized_markets` and `reference_candidates` are the curated vocabulary
        the identify step already produced, carried through with whatever the
        buyer corrected. Without them the resolve would throw that work away and
        the enrichment would pay for it again.
        """
        return await self._call(
            "POST",
            "/catalog/resolve",
            json={
                "text": text,
                "source": source,
                "brand": brand,
                "category": category,
                "localized_markets": localized_markets,
                "reference_candidates": reference_candidates,
                "target_age_min": target_age_min,
                "target_age_max": target_age_max,
            },
        )

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

    async def register_analysis(self, **payload: Any) -> dict[str, Any]:
        return await self._call("POST", "/internal/analyses", json=payload)

    async def read_analysis(self, request_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/internal/analyses/{request_id}")

    async def mark_running(self, request_id: str) -> dict[str, Any]:
        return await self._call("POST", f"/internal/analyses/{request_id}/running")

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
        self, entity_key: str, countries: list[str], *, with_candidates: bool = True
    ) -> dict[str, Any]:
        return await self._call(
            "POST",
            "/internal/enrich-terms",
            json={"entity_key": entity_key, "countries": countries, "with_candidates": with_candidates},
        )

    async def ensure_coverage(self, country_iso3: str, entity_key: str | None) -> dict[str, Any]:
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
            json={"country_iso3": country_iso3, "entity_key": entity_key},
        )

    async def candidates_coverage(
        self, entity_key: str, requested_countries: list[str], period_key: str
    ) -> dict[str, Any]:
        """Collect the reference markets, unless the period's champion is known.

        Cached means the collection is skipped: candidate demand is the slowest
        work in the system and re-running it produces the same answer.

        The stored champion comes back with the flag, and the figure matters as
        much as the fact: a stored champion that a requested market now beats
        has to be replaced, not crowned again.
        """
        return await self._call(
            "POST",
            "/internal/candidates/coverage",
            json={
                "entity_key": entity_key,
                "requested_countries": requested_countries,
                "period_key": period_key,
            },
        )

    async def demand_ceiling(
        self, entity_key: str, countries: list[str], period_key: str
    ) -> tuple[float | None, dict[str, Any] | None]:
        body = await self._call(
            "POST",
            "/internal/demand-ceiling",
            json={"entity_key": entity_key, "countries": countries, "period_key": period_key},
        )
        return body.get("demand_ceiling"), body.get("demand_ruler")

    async def score_candidates(
        self,
        entity_key: str,
        requested_countries: list[str],
        period_key: str,
        demand_ceiling: float | None,
    ) -> list[dict[str, Any]]:
        body = await self._call(
            "POST",
            "/internal/candidates/score",
            json={
                "entity_key": entity_key,
                "requested_countries": requested_countries,
                "period_key": period_key,
                "demand_ceiling": demand_ceiling,
            },
        )
        return body.get("candidates", [])

    async def set_champion(
        self, entity_key: str, period_key: str, *, country: str | None, mos: float | None
    ) -> dict[str, Any]:
        """A null country is a real outcome: "nothing was scorable this period".
        Without recording it, the same fruitless work runs on every later
        analysis."""
        return await self._call(
            "POST",
            "/internal/champion",
            json={"entity_key": entity_key, "period_key": period_key, "country": country, "mos": mos},
        )

    async def compute_indicators(
        self, entity_key: str | None, country_iso3: str, period_key: str, demand_ceiling: float | None
    ) -> dict[str, Any]:
        return await self._call(
            "POST",
            "/indicators/compute",
            json={
                "entity_key": entity_key,
                "country": country_iso3,
                "period_key": period_key,
                "demand_ceiling": demand_ceiling,
            },
        )

    async def calibrate_weights(self, period_key: str) -> dict[str, float]:
        body = await self._call("POST", "/internal/weights/calibrate", json={"period_key": period_key})
        return body.get("weights", {})

    async def compute_mos(
        self, entity_key: str, countries: list[str], period_key: str
    ) -> list[dict[str, Any]]:
        body = await self._call(
            "POST",
            "/mos/compute",
            json={"entity_key": entity_key, "countries": countries, "period_key": period_key},
        )
        return body.get("results", body if isinstance(body, list) else [])

    async def build_evidence(
        self,
        entity_key: str,
        countries: list[str],
        period_key: str,
        demand_ruler: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        body = await self._call(
            "POST",
            "/internal/evidence",
            json={
                "entity_key": entity_key,
                "countries": countries,
                "period_key": period_key,
                "demand_ruler": demand_ruler,
            },
        )
        return body.get("packet")

    async def narrate(
        self, packet: dict[str, Any], product_name: str, user_prompt: str | None = None
    ) -> str | None:
        body = await self._call(
            "POST",
            "/internal/narrate",
            json={"packet": packet, "product_name": product_name, "user_prompt": user_prompt},
        )
        return body.get("narrative")


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)[:500]
