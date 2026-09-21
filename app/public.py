"""What the browser talks to.

The agent is the way in now, not something the scoring service hands work to
after doing half of it. Before this, a request went browser -> scoring ->
agent, and the scoring service resolved the product before the agent was
involved at all — which is why a run could come back with the product named
correctly and the analysis failed, when the agent was the part that was down.

None of these endpoints decide anything. They take what the browser asked for,
call the tools, and start the run. The deciding is in the graph, which is the
point of the whole arrangement.

The data stays where it belongs: the scoring service owns the catalogue, the
analysis record and the method, and answers here over its internal API. This
service owns the run.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.scoring_client import ScoringClient, ScoringRejected, ScoringUnavailable

logger = logging.getLogger(__name__)

router = APIRouter()


def _client() -> ScoringClient:
    return ScoringClient()


def _as_http_error(exc: Exception) -> HTTPException:
    """The browser is owed the difference. A rejection is the request being
    wrong and repeating it changes nothing; an outage is worth trying again."""
    if isinstance(exc, ScoringRejected):
        return HTTPException(status_code=exc.status_code, detail=exc.detail)
    return HTTPException(status_code=503, detail=f"The scoring service is unavailable: {exc}")


@router.post(
    "/catalog/identify-image",
    status_code=200,
    summary="Identify a product from a photo or a line of text",
)
async def identify_image(payload: dict[str, Any]) -> Any:
    """Deliberately separate from POST /analyze: the buyer sees this result and
    corrects it — the name, the brand, the markets — before anything is scored."""
    try:
        return await _client().identify_image(payload)
    except (ScoringUnavailable, ScoringRejected) as exc:
        raise _as_http_error(exc) from exc


@router.get("/countries", summary="The markets that can be analysed")
async def countries() -> Any:
    try:
        return await _client().countries()
    except (ScoringUnavailable, ScoringRejected) as exc:
        raise _as_http_error(exc) from exc


@router.post("/analyze", status_code=202, summary="Analyse a product across markets")
async def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve the product, open the record, and start the run.

    Answers 202 with the id and lets the caller watch: the run takes minutes and
    a request held open for it is a request a proxy will cut.

    The run is started as a task rather than awaited. If this pod dies mid-run
    the task dies with it — and that is survivable, because the scoring service
    sweeps analyses that stopped reporting and re-invokes them by the same id,
    which is a resume rather than a second run.
    """
    product_name = (payload.get("product_name") or "").strip()
    if not product_name:
        raise HTTPException(status_code=422, detail="product_name is required")
    countries_asked = payload.get("countries") or []
    if not countries_asked:
        raise HTTPException(status_code=422, detail="countries is required: at least one market")

    core = _client()
    try:
        # The brand PARTITIONS identity: a request naming a brand resolves to
        # that brand's entity, with the generic family linked behind it.
        resolved = await core.resolve_product(
            product_name,
            source="analyze",
            brand=payload.get("brand"),
            category=payload.get("category"),
            localized_markets=payload.get("localized_markets"),
            reference_candidates=payload.get("reference_candidates"),
            target_age_min=payload.get("target_age_min"),
            target_age_max=payload.get("target_age_max"),
        )
        registered = await core.register_analysis(
            entity_key=resolved["entity_key"],
            canonical_title=resolved["canonical_title"],
            countries=countries_asked,
            estimated_price=payload.get("estimated_price"),
            price_currency=payload.get("price_currency"),
            user_prompt=payload.get("user_prompt"),
        )
    except (ScoringUnavailable, ScoringRejected) as exc:
        raise _as_http_error(exc) from exc

    request_id = str(registered["request_id"])
    _start(request_id)
    return {"request_id": request_id, "status": registered["status"]}


# Strong references to the runs in flight. asyncio keeps only a WEAK reference to
# a task, so a task nobody holds can be collected mid-run and cancelled — the
# analysis would simply stop, with the record left saying `running` until the
# sweep picked it up. Rare, silent, and impossible to reproduce on demand.
_in_flight: set[asyncio.Task[None]] = set()


def _start(analysis_id: str) -> None:
    task = asyncio.create_task(_run(analysis_id))
    _in_flight.add(task)
    task.add_done_callback(_in_flight.discard)


async def _run(analysis_id: str) -> None:
    """Drive the graph for one analysis, outside the request that asked for it.

    Every failure the graph can report, it reports on the analysis record
    itself. What is caught here is only the kind that leaves nothing written —
    and that is logged rather than raised, because there is no longer a caller
    to raise to.
    """
    from app.graph import get_graph
    from app.observability import callbacks
    from app import spend

    try:
        graph = get_graph()
    except RuntimeError:
        logger.exception("Analysis %s could not start: the graph is not built", analysis_id)
        return

    config = {"configurable": {"thread_id": analysis_id}, "callbacks": callbacks()}
    try:
        with spend.tally():
            await graph.ainvoke({"analysis_id": analysis_id}, config=config)
    except Exception:  # noqa: BLE001 — nothing left to raise to; the record carries the outcome
        logger.exception("Analysis %s ended in an unhandled error", analysis_id)
        try:
            await _client().fail_analysis(analysis_id, "The analysis ended unexpectedly.")
        except Exception:  # noqa: BLE001
            logger.exception("Analysis %s could not even be marked failed", analysis_id)


@router.get("/analyze/{request_id}", summary="Read an analysis")
async def read_analysis(request_id: str, response: Response) -> Any:
    try:
        uuid.UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="request_id must be a UUID") from exc

    # Polled while the answer is still changing, at a URL that never changes.
    # Anything in between is otherwise free to serve the first body it saw.
    response.headers["Cache-Control"] = "no-store"
    try:
        return await _client().read_report(request_id)
    except (ScoringUnavailable, ScoringRejected) as exc:
        raise _as_http_error(exc) from exc


@router.get("/analyze/{request_id}/stream", summary="Live status stream (SSE)")
async def stream_analysis(request_id: str, request: Request) -> StreamingResponse:
    """Passed through from the scoring service rather than watched again here.

    The watch is a loop over one row — which status, which progress line,
    whether the narrative has landed — and it belongs next to the row. Writing a
    second copy of it here would mean two answers to "is it finished yet", and
    they would drift.
    """
    try:
        uuid.UUID(request_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="request_id must be a UUID") from exc

    client = _client()

    async def passthrough():  # noqa: ANN202
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0)) as http:
                async with http.stream(
                    "GET",
                    f"{client.base_url}/analyze/{request_id}/stream",
                    headers=client.headers(),
                ) as upstream:
                    async for chunk in upstream.aiter_bytes():
                        if await request.is_disconnected():
                            return
                        yield chunk
        except httpx.HTTPError as exc:
            # Said on the stream, not swallowed. A stream that closes quietly
            # reads to the browser exactly like one that finished.
            logger.warning("The stream for %s dropped: %s", request_id, exc)
            yield b'data: {"status": "stream_error"}\n\n'

    return StreamingResponse(
        passthrough(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
