import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.graph import build_graph

settings = get_settings()

app = FastAPI(title="MarketMind Orchestrator", root_path=settings.root_path)

graph = build_graph()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/run-analysis")
async def run_analysis(request: Request):
    payload = await request.json()

    async def event_generator():
        # A run that dies mid-stream used to close the connection with no final
        # event, which reads to the caller exactly like a run that finished
        # quietly. The caller then has to guess, and the guess that costs money
        # is "it worked". So a failure says so, on the stream, before closing.
        try:
            async for event in graph.astream(payload):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001 — the stream is the only channel back
            yield f"data: {json.dumps({'error': str(exc), 'type': type(exc).__name__})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
