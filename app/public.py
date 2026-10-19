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
import os
import socket
import uuid
from functools import lru_cache
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
            # Como o catálogo chegou nesta entidade, guardado com a análise.
            # Aqui ainda não há trace do Langfuse — ele nasce quando o grafo
            # roda, adiante — e uma retomada nunca repete esta resolução. Sem
            # gravar, o fato se perde nos dois casos.
            catalog_method=resolved.get("method"),
            catalog_match_score=resolved.get("match_score"),
            catalog_match_reasoning=resolved.get("match_reasoning"),
            # O que o comprador digitou, guardado ao lado do que o catálogo
            # respondeu: sem o par, uma resolução não é auditável.
            catalog_query_text=product_name,
            # O vínculo da marca, que é metade da resposta: a análise fica
            # ligada ao produto E à marca, e nulo aqui significa "foi do
            # produto" — é isso que separa uma coisa da outra depois.
            brand_key=resolved.get("brand_key"),
            brand_name=resolved.get("brand"),
            countries=countries_asked,
            estimated_price=payload.get("estimated_price"),
            price_currency=payload.get("price_currency"),
            user_prompt=payload.get("user_prompt"),
        )
    except (ScoringUnavailable, ScoringRejected) as exc:
        raise _as_http_error(exc) from exc

    request_id = str(registered["request_id"])
    # Claimed before it starts, so a restart elsewhere does not pick it up while
    # this process is working on it.
    try:
        await core.claim_analysis(request_id, runner=runner_id())
    except (ScoringUnavailable, ScoringRejected) as exc:
        logger.warning("Analysis %s could not be claimed; running it anyway: %s", request_id, exc)
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


@lru_cache(maxsize=1)
def runner_id() -> str:
    """Who this process is, for claiming an analysis. The pod name in the
    cluster; the hostname anywhere else. It only has to be different from the
    other runners, and stable for as long as this process lives."""
    return os.getenv("HOSTNAME") or socket.gethostname()


# How often to look for analyses nobody is running. Asking once at startup is
# not enough and the reason is worth stating, because it was measured rather
# than reasoned: a pod killed mid-analysis is replaced in about fifteen seconds,
# and the claim it held is good for two minutes. So the replacement starts,
# finds nothing orphaned — the dead pod's claim has not expired yet — and never
# asks again. The orphan then waits for the fifteen-minute sweep, which is
# exactly the wait this was built to remove.
#
# Slightly longer than the claim's own staleness window, so a claim that is
# about to expire is not chased before it has.
_ORPHAN_CHECK_SECONDS = 150


async def watch_for_orphans() -> None:
    """Keep picking up analyses that nobody is running, for as long as this
    process lives. Never raises: it must outlive every individual failure."""
    while True:
        try:
            await resume_orphans()
        except Exception:  # noqa: BLE001 — the watch must survive anything a single pass hits
            logger.exception("The orphan check failed; trying again next time")
        await asyncio.sleep(_ORPHAN_CHECK_SECONDS)


async def resume_orphans() -> int:
    """Pick up the analyses that are running with nobody running them.

    This is one pass. It runs at startup — where it catches what a predecessor
    that died long enough ago left behind — and then on a timer, which is what
    catches the common case: the pod is replaced in seconds and the claim it
    held is still good for another two minutes.

    Claiming is what makes it safe with more than one replica: an analysis
    another agent is actively running is not orphaned, and the claim it holds is
    renewed on a clock for as long as the run lasts.

    Never raises. A scoring service that is not up yet resumes nothing, and the
    sweep remains what it always was — the net under this.
    """
    try:
        orphans = await _client().orphaned_analyses()
    except (ScoringUnavailable, ScoringRejected) as exc:
        logger.warning("Could not ask which analyses were orphaned, so none were resumed: %s", exc)
        return 0

    resumed = 0
    for analysis_id in orphans:
        try:
            if not await _client().claim_analysis(analysis_id, runner=runner_id()):
                continue  # someone live has it
        except (ScoringUnavailable, ScoringRejected) as exc:
            logger.warning("Could not claim analysis %s, leaving it to the sweep: %s", analysis_id, exc)
            continue
        logger.info("Resuming analysis %s, orphaned by a restart", analysis_id)
        _start(analysis_id)
        resumed += 1

    if orphans:
        logger.info("Startup: %s of %s orphaned analyses resumed here", resumed, len(orphans))
    return resumed


# A claim is a LEASE, and a lease is renewed on a clock. Renewing it on progress
# reports alone was wrong: a single collection step can run for minutes without
# reporting anything, and a claim that ages out mid-step tells another agent the
# analysis is orphaned while it is very much alive. Half the staleness window,
# so two renewals have to be missed before anyone else may take over.
_CLAIM_RENEWAL_SECONDS = 45


async def _hold_claim(analysis_id: str) -> None:
    """Keep saying "this one is mine" for as long as the run lasts."""
    core = _client()
    while True:
        await asyncio.sleep(_CLAIM_RENEWAL_SECONDS)
        try:
            await core.claim_analysis(analysis_id, runner=runner_id())
        except Exception:  # noqa: BLE001 — a missed renewal is not a reason to stop the run
            logger.warning("Could not renew the claim on analysis %s", analysis_id, exc_info=True)


async def _run(analysis_id: str) -> None:
    """Drive the graph for one analysis, outside the request that asked for it.

    Every failure the graph can report, it reports on the analysis record
    itself. What is caught here is only the kind that leaves nothing written —
    and that is logged rather than raised, because there is no longer a caller
    to raise to.
    """
    from app.graph import get_graph
    from app.observability import callbacks, run_span
    from app import metrics, spend

    try:
        graph = get_graph()
    except RuntimeError:
        logger.exception("Analysis %s could not start: the graph is not built", analysis_id)
        return

    config = {"configurable": {"thread_id": analysis_id}, "callbacks": callbacks()}
    lease = asyncio.create_task(_hold_claim(analysis_id))
    try:
        # O span aberto aqui é o que dá contexto a TUDO que roda dentro: sem
        # ele os scores eram descartados em silêncio e os fatos viravam traces
        # soltos. Ver observability.run_span.
        with spend.tally(), run_span(f"analysis {analysis_id}", seed=analysis_id):
            await graph.ainvoke({"analysis_id": analysis_id}, config=config)
            # Depois do grafo INTEIRO, e não na entrega. A narrativa vem depois
            # do deliver e é a maior chamada de modelo da execução; medindo no
            # deliver, o total saía sem ela, sem o juiz e sem a revisão. Aqui
            # dentro de propósito: o span e a contagem ainda estão abertos.
            metrics.token_consumption()
            metrics.demand_cost()
            metrics.analysis_cost(analysis_id)
    except Exception:  # noqa: BLE001 — nothing left to raise to; the record carries the outcome
        logger.exception("Analysis %s ended in an unhandled error", analysis_id)
        try:
            await _client().fail_analysis(analysis_id, "The analysis ended unexpectedly.")
        except Exception:  # noqa: BLE001
            logger.exception("Analysis %s could not even be marked failed", analysis_id)
    finally:
        # Released as soon as the run ends, however it ends. A lease still being
        # renewed after the work stopped is the one thing that would keep an
        # orphan invisible.
        lease.cancel()


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
