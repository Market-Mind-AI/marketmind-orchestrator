import asyncio
import hmac
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app import spend
from app.checkpointer import open_checkpointer
from app.config import get_settings
from app.graph import build_graph, set_graph
from app.observability import callbacks, tracing_enabled
from app.public import router as public_router, watch_for_orphans

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
    service's resume path starts runs here, and a run costs money. Until now
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
    payload = await request.json()

    # The thread id IS the analysis id, and that is what makes re-invoking mean
    # "resume" instead of "start another one". A generated id would create a
    # second run of the same analysis every time anything retried.
    analysis_id = payload.get("analysis_id")
    if not analysis_id:
        raise HTTPException(status_code=422, detail="analysis_id is required: it is the run's identity.")

    config = {"configurable": {"thread_id": str(analysis_id)}, "callbacks": callbacks()}

    async def event_generator():
        # A run that dies mid-stream used to close the connection with no final
        # event, which reads to the caller exactly like a run that finished
        # quietly. The caller then has to guess, and the guess that costs money
        # is "it worked". So a failure says so, on the stream, before closing.
        #
        # The tally is opened around the whole stream so every node's tool calls
        # add to the same one; the nodes read it when they report the run.
        try:
            with spend.tally():
                async for event in request.app.state.graph.astream(payload, config=config):
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001 — the stream is the only channel back
            logger.exception("Analysis %s failed", analysis_id)
            yield f"data: {json.dumps({'error': str(exc), 'type': type(exc).__name__})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
