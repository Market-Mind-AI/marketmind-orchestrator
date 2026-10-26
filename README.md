# MarketMind Orchestrator

The agent that runs one MarketMind analysis end to end. A LangGraph graph, served
by FastAPI, that drives the method by calling the scoring service's `/internal`
tools — it decides *what to do next*, and computes nothing itself.

It is the entry point the frontend talks to. The scoring service
(`marketmind-scoring`, still deployed under the name `marketmind-core`) owns the
method, the data and the records; the four collectors and the Ollama gateway sit
behind its tools. This service knows only one address: the scoring service's.

## What a run is

```
start → enrich → collect → ruler → measure → score → champion → evidence → check → deliver → narrate → END
                                                  │                             │
           ruler / measure / score could not finish, a collection failed,      │
           or the result does not hold together  ──────────►  decide (agent) ◄──┘
                                                                  │
            retry_later → collect · wait_and_resume → wait → collect · recompute_once → ruler
            continue_without / deliver_with_note → champion (or deliver) · deliver → deliver
            give_up → give_up → END
```

- **start** fixes the run's identity: the `period_key` (decided once and sent on
  every scoring call that takes it), the clock the latency is read from, and the
  24 h deadline, which is also sent to the scoring service (`/running`).
- **enrich** resolves the search terms per country (brand and generic vocabulary).
- **collect** asks the scoring service which markets had no demand recently
  (`/markets/demand-absence`) and collects the rest.
- **ruler** reads the demand ceiling the countries are measured against.
- **measure** and **score** compute the indicators and the MOS — in the scoring
  service, not here. A timeout or 5xx there goes to `decide`, not to failure.
- **champion** asks the scoring service to crown the product's best market
  (`/champion/crown`) and delivers the champion's row when nobody asked for it;
  **evidence** fetches the packet every figure in the narrative must come from;
  **check** compares the scoring service's answers with each other (a requested
  market missing from the evidence, a champion without its row or evidence, a
  ruler that moved after measuring); **deliver** records the result as scoring
  made it; **narrate** writes the explanation and runs the quality checks.
- **decide** is the agent (`app/agent.py`): when a collection failed after the
  scoring service's retries, a source or the scoring service is down, or the
  result is incoherent, a model picks one action from a closed set the code
  computes from the state — `retry_later`, `wait_and_resume`,
  `continue_without`, `deliver_with_note`, `recompute_once`, `deliver`,
  `give_up` — and gives a reason, recorded in the log, the trace and the
  progress line. Each action runs through the nodes above; the model never
  produces a number. Without `ANTHROPIC_API_KEY` the fixed rule decides
  (insist up to 3 rounds within the deadline, then give up).
- **give_up** fails the analysis with a reason. A market the sources simply do
  not hold ends the run here directly.

Every node calls a tool and records what it learned. The MOS, the indicators,
the champion and the evidence stay where the method is versioned.

## Durability

An analysis can run for hours and may be resumed after the pod that started it
is gone. The **checkpointer** (`langgraph-checkpoint-postgres`) writes a
snapshot after each step, keyed by the analysis id; re-invoking the graph with
the same id resumes from the last completed node — every step is idempotent, so
resuming is re-running.

Every run starts through one path (`public._run`): claim the analysis, renew
the claim as a lease while it runs, report its cost at the end. `/analyze`, the
orphan watch (every 150 s, in every replica) and `/run-analysis` all go through
it, so one analysis never executes twice. A run that already delivered is never
marked failed afterwards.

LangGraph creates its tables itself on startup; there is no migration step.
`CHECKPOINTER=memory` keeps it in-process and loses it on restart — it is for
tests and has to be asked for by name.

## Routes

| Method | Path | What it does |
|---|---|---|
| `GET` | `/healthz` | Liveness. |
| `POST` | `/run-analysis` | Runs (or resumes) the graph for `analysis_id` through the same claim as every other start, streaming graph events as SSE (`skipped` when another runner holds it). Requires the shared secret as `X-Internal-Token`. |
| `POST` | `/catalog/identify-image` | Identifies a product from a photo or a line of text, so the buyer can correct it before anything is scored. |
| `GET` | `/countries` | The markets that can be analysed. |
| `POST` | `/analyze` | Registers the analysis through the scoring service (`/internal/analyses/register`, which resolves the product) and starts the run. Answers `202` with the id. |
| `GET` | `/analyze/{id}` | The analysis as it stands (`Cache-Control: no-store`). |
| `GET` | `/analyze/{id}/stream` | Live status, passed through from the scoring service's SSE stream. |

Behind a reverse proxy, set `ROOT_PATH` (e.g. `/orchestrator`) so the Swagger
page builds its URLs under the right prefix.

## Quality controls

The narrative is written from a deterministic evidence packet, so a figure that
is not in the packet is not supported by anything. `app/quality.py` checks:

- `unsupported_numbers` — every number in the text, matched against the packet
  (exact, with a small tolerance for rounding);
- `tool_call_faults` — redundant and off-request tool calls.

The phrases the method forbids ("proven demand", "guarantee") are checked by the
scoring service (`/narrative/language-check`), in the narrative's language.
Sentences that claim something without a figure are the Langfuse evaluator's.

## Measures and cost

`app/metrics.py` gives each measure one definition, logs it as a `QUALITY` line
and attaches it to the run's trace as a score:

`task_completion_rate`, `response_latency_seconds`, `token_consumption`,
`dataforseo_cost_usd`, `analysis_cost_usd`, `tool_invocation_accuracy`,
`hallucination_rate`.

The run also writes one `COST` line — total dollars, models by name, and what
DataForSEO billed for the demand data:

```
COST analysis=<id> total_usd=1.64 models_usd=0.56 (claude-opus-5 0.50, gpt-5.1 0.05) dataforseo_usd=1.08 (12 tasks)
```

Model calls are priced by the table Langfuse holds (`app/pricing.py`, read at
startup from a thread and retried until it loads); the scoring service prices
its own calls the same way and reports them on response headers, which
`app/spend.py` tallies per run. A call that cannot be priced is said, not zeroed.

## Observability

Every run is one Langfuse trace, seeded by the analysis id so it can be reached
from the id the rest of the system already has. The `traceparent` header is sent
to the scoring service on every call, so the model calls it makes land inside
the same trace. `observability.flag()` marks the run's root span `WARNING` or
`ERROR` when the graph fails on purpose (a `give_up`, a part that did not
finish), so the Quality view lists it. Unconfigured Langfuse means untraced,
said once at startup.

## Configuration

From `app/config.py` (read from `.env` and the environment; unknown variables
are ignored):

| Variable | Default | Notes |
|---|---|---|
| `ROOT_PATH` | `` | Proxy prefix. |
| `LOG_LEVEL` | `INFO` | |
| `SCORING_API_URL` | `http://marketmind-core:8000` | The only service this one calls. |
| `SCORING_INTERNAL_TOKEN` | — | Required by the scoring service's `/internal` routes, and required back on `POST /run-analysis` (`X-Internal-Token`; unset closes the route with 503). The scoring service holds the same value as `INTERNAL_API_TOKEN`. |
| `CHECKPOINTER` | `postgres` | `memory` only for tests. |
| `CHECKPOINTER_DB_HOST/PORT/NAME/USER/PASSWORD` | — | The service's own database. |
| `CHECKPOINTER_DATABASE_URL` | — | Wins over the parts when set. |
| `SCORING_COLLECTION_TIMEOUT_FLOOR_SECONDS` | `2100` | Collecting calls wait until the run's deadline, never less than this (one scoring gap is ~1952 s). |
| `SCORING_COLLECTION_TIMEOUT_MAX_SECONDS` | `86400` | What a collecting call waits outside a run. |
| `ANTHROPIC_API_KEY` | — | The agent's decision model. Unset (or the `anthropic` package missing) means the fixed rule decides. |
| `DECISION_MODEL` | `claude-sonnet-5-5` | |
| `SOURCE_WAIT_SECONDS` | `1800` | One `wait_and_resume`, at most; never past the deadline. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | — | Unset means untraced. |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | |

`docker-compose.yml` brings up the API and its Postgres (host port 5437).

## Running

```bash
poetry install
cp .env.example .env
docker compose up -d postgres
uvicorn app.main:app --reload --port 8081
```

Tests (the "Failed to export span batch 404" lines are the OTLP exporter with
no collector, and harmless):

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
```

## Documents

- `docs/superpowers/specs/2026-09-23-agent-memory-design.md` — the store of
  barren markets, kept as history: the scoring service now answers that from
  the data.
- `docs/superpowers/specs/2026-08-04-marketmind-orchestrator-design.md` and
  `docs/superpowers/plans/2026-08-04-orchestrator-mvp.md` — the original
  design, kept as history; the banner at the top of each says what changed.

## Team

Final project (TFM) for the *Máster en Ingeniería y Desarrollo de Soluciones de IA Generativa* — **EBIS Business Techschool**. The whole system and how to run it: [marketmind-ai](https://github.com/Market-Mind-AI/marketmind-ai).

- **Miguel Antonio Blanco Salcedo** — [@mikews93](https://github.com/mikews93)
- **Álvaro Augusto De Marco Neto** — [@ironeto](https://github.com/ironeto)
