# Orchestrator MVP Implementation Plan

> **HISTÓRICO / SUPERADO.** Este documento é o desenho de origem (04/08/2026) e não descreve
> mais o serviço. O que vale hoje está no `README.md` deste repositório e nas specs posteriores em
> `docs/superpowers/specs/*`. O que mudou desde então:
>
> - **Sete rotas, não uma.** Além de `POST /run-analysis` (hoje chamado pelo caminho de retomada do
>   serviço de scoring), o orquestrador é a porta de entrada do frontend: `POST /analyze` (202),
>   `GET /analyze/{id}`, `GET /analyze/{id}/stream`, `POST /catalog/identify-image`,
>   `GET /countries`, `GET /healthz`.
> - **Nenhum nó de LLM e nenhum fan-out direto aos coletores.** Os 11 nós chamam apenas as
>   ferramentas `/internal` do serviço de scoring; extração, coleta, MOS e narrativa acontecem lá.
>   Este serviço só decide o próximo passo.
> - **Não é mais stateless.** Checkpointer e store do LangGraph em Postgres (banco próprio): uma
>   análise retoma do último nó concluído e o agente lembra, entre execuções, os mercados sem dados.
> - **O scoring é chamado por HTTP**, com token interno e `traceparent`, e não importado como
>   biblioteca nem embutido no grafo.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a stateless FastAPI + LangGraph microservice that orchestrates the MarketMind enrichment and scoring pipeline.

**Architecture:** A FastAPI app exposing `POST /run-analysis` that streams progress via SSE. The pipeline is a LangGraph `StateGraph` consisting of extraction, parallel enrichment, scoring, geo matching, and reporting nodes.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, Pydantic, HTTPX, Pytest, Uvicorn.

## Global Constraints
- Python 3.12+
- Use `httpx` for all external calls.
- Configuration injected via environment variables (`pydantic-settings`).
- Complete statelessness (no database).
- Adhere to the defined LangGraph state schema.

---

### Task 1: Project Setup and Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: Environment variables
- Produces: Base FastAPI app, configuration settings

- [ ] **Step 1: Write configuration tests**
```python
def test_config_loads_from_env(monkeypatch):
    monkeypatch.setenv("WORLDBANK_API_URL", "http://wb")
    from app.config import get_settings
    settings = get_settings()
    assert settings.worldbank_api_url == "http://wb"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_config.py -v`
Expected: FAIL

- [ ] **Step 3: Implement minimal configuration**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    worldbank_api_url: str = "http://marketmind-worldbank:8000"
    oecd_api_url: str = "http://marketmind-oecd:8000"
    un_api_url: str = "http://marketmind-united-nations-data:8000"
    pytrends_api_url: str = "http://marketmind-pytrends:8000"
    llm_api_url: str = "http://marketmind-ollma-llm:8000"
    
def get_settings():
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Setup minimal FastAPI app**
```python
from fastapi import FastAPI
app = FastAPI(title="MarketMind Orchestrator")
@app.get("/healthz")
def healthz():
    return {"status": "ok"}
```

- [ ] **Step 6: Commit**
```bash
git add pyproject.toml app/ tests/
git commit -m "chore: setup poetry project and fastapi base"
```

### Task 2: State Definition

**Files:**
- Create: `app/state.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Consumes: N/A
- Produces: `OrchestratorState` typed dict for LangGraph

- [ ] **Step 1: Write test for state updates**
```python
def test_state_schema():
    from app.state import OrchestratorState
    state: OrchestratorState = {"analysis_id": "123", "errors": []}
    assert state["analysis_id"] == "123"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_state.py -v`
Expected: FAIL

- [ ] **Step 3: Implement state schema**
```python
from typing import TypedDict, Annotated, List
import operator

class OrchestratorState(TypedDict):
    analysis_id: str
    product_image_url: str
    product_description: str
    attributes: dict
    enrichment_results: Annotated[List[dict], operator.add]
    mos_score: dict
    report: dict
    errors: Annotated[List[str], operator.add]
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/state.py tests/test_state.py
git commit -m "feat: define langgraph state schema"
```

### Task 3: Connector Interfaces

**Files:**
- Create: `app/connectors.py`
- Create: `tests/test_connectors.py`

**Interfaces:**
- Consumes: `app.config.Settings`
- Produces: Async functions to fetch data from the 4 microservices.

- [ ] **Step 1: Write test for connectors (mocked HTTPX)**
```python
import pytest
import httpx
from app.connectors import fetch_worldbank_data

@pytest.mark.asyncio
async def test_fetch_worldbank(respx_mock):
    respx_mock.get("http://marketmind-worldbank:8000/api/data?q=test").mock(return_value=httpx.Response(200, json={"data": "wb"}))
    res = await fetch_worldbank_data("test")
    assert res == {"source": "worldbank", "data": {"data": "wb"}}
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_connectors.py -v`
Expected: FAIL

- [ ] **Step 3: Implement HTTPX clients**
```python
import httpx
from app.config import get_settings

async def fetch_worldbank_data(product: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{settings.worldbank_api_url}/api/data", params={"q": product})
        return {"source": "worldbank", "data": resp.json()}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_connectors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/connectors.py tests/test_connectors.py
git commit -m "feat: implement enrichment connectors"
```

### Task 4: LangGraph Nodes Implementation

**Files:**
- Create: `app/nodes.py`
- Create: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `app.connectors`, `app.state.OrchestratorState`
- Produces: Individual functions representing graph nodes (extract, enrich, score, report).

- [ ] **Step 1: Write node tests**
```python
import pytest
from app.nodes import scoring_node

def test_scoring_node():
    state = {"enrichment_results": [{"data": 10}]}
    result = scoring_node(state)
    assert "mos_score" in result
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_nodes.py -v`
Expected: FAIL

- [ ] **Step 3: Implement nodes**
```python
async def extraction_node(state: dict):
    # Stub: Call Haiku Vision LLM
    return {"attributes": {"category": "stub"}}

async def enrichment_worldbank_node(state: dict):
    # Call fetch_worldbank_data
    return {"enrichment_results": [{"source": "wb"}]}

def scoring_node(state: dict):
    # Pure Python MOS arithmetic
    return {"mos_score": {"final_score": 85}}

async def report_node(state: dict):
    # Stub: Call marketmind-ollma-llm
    return {"report": {"narrative": "Stub report"}}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_nodes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/nodes.py tests/test_nodes.py
git commit -m "feat: implement graph nodes"
```

### Task 5: Graph Assembly

**Files:**
- Create: `app/graph.py`
- Create: `tests/test_graph.py`

**Interfaces:**
- Consumes: `app.nodes`, `app.state`
- Produces: `CompiledGraph` object

- [ ] **Step 1: Write graph execution test**
```python
import pytest
from app.graph import build_graph

@pytest.mark.asyncio
async def test_graph_execution():
    graph = build_graph()
    result = await graph.ainvoke({"analysis_id": "123"})
    assert "mos_score" in result
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_graph.py -v`
Expected: FAIL

- [ ] **Step 3: Implement graph construction**
```python
from langgraph.graph import StateGraph, END
from app.state import OrchestratorState
from app.nodes import extraction_node, enrichment_worldbank_node, scoring_node, report_node

def build_graph():
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("extract", extraction_node)
    workflow.add_node("enrich_wb", enrichment_worldbank_node)
    workflow.add_node("score", scoring_node)
    workflow.add_node("report", report_node)
    
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "enrich_wb")
    workflow.add_edge("enrich_wb", "score")
    workflow.add_edge("score", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_graph.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/graph.py tests/test_graph.py
git commit -m "feat: assemble langgraph workflow"
```

### Task 6: FastAPI SSE Endpoint

**Files:**
- Modify: `app/main.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `app.graph.build_graph`
- Produces: `POST /run-analysis` streaming endpoint (using FastAPI `StreamingResponse`)

- [ ] **Step 1: Write API test**
```python
from fastapi.testclient import TestClient
from app.main import app

def test_run_analysis_endpoint():
    client = TestClient(app)
    response = client.post("/run-analysis", json={"analysis_id": "123"})
    assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_api.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SSE Endpoint**
```python
from fastapi import Request
from fastapi.responses import StreamingResponse
from app.graph import build_graph
import json

graph = build_graph()

@app.post("/run-analysis")
async def run_analysis(request: Request):
    payload = await request.json()
    
    async def event_generator():
        async for event in graph.astream(payload):
            yield f"data: {json.dumps(event)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add SSE streaming endpoint for analysis execution"
```
