# MarketMind Orchestrator - Design Spec

> **A direção da chamada inverteu (nota de 23/09/2026).** A seção 4 descreve o `marketmind-core`
> invocando o orquestrador e repassando o SSE ao frontend. É o contrário do que está no ar desde
> 21/09/2026: **o orquestrador é a porta de entrada** — o frontend fala com ele em `/api`, e é ele que
> chama o serviço de método, renomeado de `marketmind-core` para `marketmind-scoring`, pelas rotas
> `/internal`. A spec fica como está porque é o desenho de origem; o que vale hoje está no
> `README.md` deste repositório.


## 1. Overview
The `marketmind-orchestrator` is a dedicated, stateless, cloud-agnostic microservice responsible for orchestrating the MarketMind analysis pipeline. It executes a LangGraph `StateGraph` (DAG) that integrates vision attribute extraction, parallel data enrichment, deterministic Market Opportunity Score (MOS) calculation, and LLM-based narrative generation.

## 2. Architecture & Tech Stack
* **Language/Framework:** Python 3.12, FastAPI
* **Orchestration:** LangGraph (`StateGraph`)
* **Concurrency:** `httpx` for asynchronous HTTP requests to external microservices.
* **API Interface:** Exposes a single `POST /run-analysis` endpoint that accepts analysis requests and streams back Server-Sent Events (SSE) representing the live progress of the LangGraph pipeline.

## 3. The LangGraph DAG (Data Flow)
The orchestrator's state graph consists of the following sequential and parallel nodes:
1. **Extract Node:** Receives product images/descriptions and calls Claude Haiku (or Ollama fallback) to extract structured product attributes.
2. **Enrichment Fan-Out Node:** Uses parallel execution to simultaneously request data from 4 connectors:
   - `marketmind-worldbank`
   - `marketmind-oecd`
   - `marketmind-united-nations-data`
   - `marketmind-pytrends`
3. **Scoring Node:** A pure Python node that executes the deterministic MOS calculation (winsorize -> min-max -> weighted sum) based on the enriched data.
4. **Geo Match Node:** Ranks geographic markets based on scoring and data signals.
5. **Report Node:** Calls `marketmind-ollma-llm` with the computed MOS evidence packet to generate a final, cited business narrative.

## 4. Integration with `marketmind-core`
* `marketmind-core` remains the CRUD backend. It handles database operations, saves uploaded images, and generates an `analysis_id`.
* Upon a user request, `marketmind-core` invokes the orchestrator's `POST /run-analysis` endpoint, passing necessary context (image URLs, descriptions, target markets).
* The orchestrator runs the LangGraph DAG and streams execution progress back to `marketmind-core` via SSE.
* `marketmind-core` proxies this SSE stream directly to the frontend (`marketmind-ai`) to provide users with a live, step-by-step progress view without intermediate polling.

## 5. Deployment & Cloud Agnosticism
* The repository will be containerized (Docker) and fully stateless.
* Configuration (URLs for the 4 connectors, LLM endpoints, credentials) will be injected via environment variables.
* The orchestrator relies on no internal database, making it easily deployable across any container orchestration platform (e.g., Kubernetes, Cloud Run, ECS) via the existing Terraform modular structure.
