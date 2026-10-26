import asyncio
import hmac
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.checkpointer import open_checkpointer
from app.config import get_settings
from app.graph import build_graph, set_graph
from app.observability import tracing_enabled
from app.public import _start as start_run, router as public_router, watch_for_orphans

from app.logging_setup import setup_logging

# Before anything else logs. Without it uvicorn configures only its own loggers
# and every line this codebase writes is dropped.
setup_logging()

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # The checkpointer holds a connection pool, so it lives as long as the
    # application does and the graph is compiled around it once.
    async with open_checkpointer() as checkpointer:
        app.state.graph = build_graph(checkpointer)
        # Also parked at module level: a run started by POST /analyze outlives
        # the request that asked for it and cannot read app.state.
        set_graph(app.state.graph)
        logger.info("Checkpointer: %s", settings.checkpointer)
        # Keep picking up analyses nobody is running: once now, then on a timer.
        # A task, not an await — a slow first pass must not hold the port closed,
        # and the runs are background work either way.
        app.state.orphan_watch = asyncio.create_task(watch_for_orphans())

        if not tracing_enabled():
            # Said once, at startup. Discovering it while trying to explain a
            # bad run is discovering it too late.
            logger.warning("Langfuse is not configured; runs will not be traced.")
        yield
        app.state.orphan_watch.cancel()


app = FastAPI(title="MarketMind Orchestrator", root_path=settings.root_path, lifespan=lifespan)


app.include_router(public_router)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def require_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    """Refuse unless the caller carries the shared secret.

    The same secret this service sends the scoring service on every /internal
    call, required back on the one call that goes the other way: the scoring
    service can start runs here, and a run costs money. Until now
    anyone who could reach this port could start one.

    With no secret configured every call is refused rather than allowed — the
    two look the same from outside a running deployment, and only one of them
    is safe to get wrong. Compared with hmac.compare_digest, so a wrong token
    takes the same time to reject whatever its prefix.
    """
    if not settings.scoring_internal_token:
        raise HTTPException(
            status_code=503,
            detail="SCORING_INTERNAL_TOKEN is not configured, so /run-analysis is closed.",
        )
    if x_internal_token is None or not hmac.compare_digest(x_internal_token, settings.scoring_internal_token):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Internal-Token.")


@app.post("/run-analysis", dependencies=[Depends(require_internal_token)])
async def run_analysis(request: Request):
    """Run (or resume) an analysis by id, streaming its graph events.

    Through the same door as every other start — claim, lease, cost report —
    so a caller that re-invokes an analysis another runner is working on gets
    told so instead of a second execution. The run is a task of its own: a
    caller that hangs up stops hearing the events, not the analysis.
    """
    payload = await request.json()

    # The thread id IS the analysis id, and that is what makes re-invoking mean
    # "resume" instead of "start another one". A generated id would create a
    # second run of the same analysis every time anything retried.
    analysis_id = payload.get("analysis_id")
    if not analysis_id:
        raise HTTPException(status_code=422, detail="analysis_id is required: it is the run's identity.")

    events: asyncio.Queue[dict | None] = asyncio.Queue()
    start_run(str(analysis_id), events.put_nowait)

    async def event_generator():
        # A run that dies mid-stream used to close the connection with no final
        # event, which reads to the caller exactly like a run that finished
        # quietly. The run says so on the stream — an `error`, or `skipped`
        # when another runner holds it — before the stream closes.
        while (event := await events.get()) is not None:
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
