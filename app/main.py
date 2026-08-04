from fastapi import FastAPI

app = FastAPI(title="MarketMind Orchestrator")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
