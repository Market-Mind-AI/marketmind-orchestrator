from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from app.graph import build_graph
import json

app = FastAPI(title="MarketMind Orchestrator")

graph = build_graph()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/run-analysis")
async def run_analysis(request: Request):
    payload = await request.json()
    
    async def event_generator():
        async for event in graph.astream(payload):
            yield f"data: {json.dumps(event)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")
