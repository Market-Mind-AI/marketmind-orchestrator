import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from app import observability, spend
from app.checkpointer import open_checkpointer, open_store
from app.config import get_settings
from app.graph import build_graph, set_graph
from app.observability import callbacks, tracing_enabled
from app.public import router as public_router

from app.logging_setup import setup_logging

# Before anything else logs. Without it uvicorn configures only its own loggers
# and every line this codebase writes is dropped.
setup_logging()

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # The checkpointer holds a connection pool, so it lives as long as the
    # application does and the graph is compiled around it once. The store
    # rides alongside it, opened from the same connection: the agent's memory
    # of past runs is meant to live exactly as long as its memory of this one.
    async with open_checkpointer() as checkpointer, open_store() as store:
        app.state.graph = build_graph(checkpointer, store)
        # Also parked at module level: a run started by POST /analyze outlives
        # the request that asked for it and cannot read app.state.
        set_graph(app.state.graph)
        logger.info("Checkpointer: %s", settings.checkpointer)
        if not tracing_enabled():
            # Said once, at startup. Discovering it while trying to explain a
            # bad run is discovering it too late.
            logger.warning("Langfuse is not configured; runs will not be traced.")
        yield


app = FastAPI(title="MarketMind Orchestrator", root_path=settings.root_path, lifespan=lifespan)


app.include_router(public_router)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/run-analysis")
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
            with observability.run(str(analysis_id)), spend.tally():
                async for event in request.app.state.graph.astream(payload, config=config):
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001 — the stream is the only channel back
            logger.exception("Analysis %s failed", analysis_id)
            yield f"data: {json.dumps({'error': str(exc), 'type': type(exc).__name__})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
