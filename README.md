# MarketMind Orchestrator

The agent that runs one MarketMind analysis end to end. A LangGraph graph, served
by FastAPI, that drives the method by calling the scoring service's `/internal`
tools — it decides *what to do next*, and computes nothing itself.

It is the entry point the frontend talks to. The scoring service
(`marketmind-scoring`, still deployed under the name `marketmind-core`) owns the
method, the data and the records; the four collectors and the Ollama gateway sit
behind its tools. This service knows only one address: the scoring service's.

## What a run is

Eleven nodes, in this order:

```
start → enrich → collect → ruler → measure → score ─┬─ complete → champion → evidence → deliver → narrate → END
                                                     ├─ insist   → collect  (back for the countries still short of data)
                                                     └─ give_up  → END      (deadline or attempts exhausted, with a reason)
```

- **start** fixes the run's identity: the `period_key` (decided once, so a run
  that crosses midnight keeps measuring the same month), the clock the latency
  is read from, and the countries asked for.
- **enrich** resolves the search terms per country (brand and generic vocabulary).
- **collect** asks the scoring service to ensure coverage for every requested
  country, consulting the agent's own memory of barren markets first.
- **ruler** reads the demand ceiling the countries are measured against.
- **measure** and **score** compute the indicators and the MOS — in the scoring
  service, not here.
- **champion** finds or refreshes the period's best market for comparison;
  **evidence** builds the packet every figure in the narrative must come from;
  **deliver** records the result; **narrate** writes the explanation and runs
  the quality checks on it.
- **insist** is not a node but the edge back to `collect`: a source that is
  down is retried for up to 24 hours and at most 3 rounds, because a country
  missing from a comparison is a different answer, not a smaller one.
- **give_up** fails the analysis with a reason rather than delivering part of a
  ranking.

Every node calls a tool and records what it learned. The MOS, the indicators
and the evidence stay where the method is versioned.

## Durability

An analysis can run for hours and may be resumed after the pod that started it
is gone. Two things therefore live in Postgres, in this service's own database:

- the **checkpointer** (`langgraph-checkpoint-postgres`): a snapshot after each
  step, keyed by the analysis id. Re-invoking the graph with the same id resumes
  from the last completed node — every step is idempotent, so resuming is
  re-running. The scoring service sweeps analyses that stopped reporting and
  re-invokes them; this service also looks for orphaned analyses on a timer.
- the **store** (`AsyncPostgresStore`): the agent's long-term memory across runs,
  today the markets found barren for a product, trusted for `MEMORY_TTL_DAYS`.

LangGraph creates both sets of tables itself on startup; there is no migration
step. `CHECKPOINTER=memory` keeps both in-process and loses them on restart —
it is for tests and has to be asked for by name.

## Routes

| Method | Path | What it does |
|---|---|---|
| `GET` | `/healthz` | Liveness. |
| `POST` | `/run-analysis` | Runs (or resumes) the graph for `analysis_id`, streaming graph events as SSE. Called by the scoring service's resume path. |
| `POST` | `/catalog/identify-image` | Identifies a product from a photo or a line of text, so the buyer can correct it before anything is scored. |
| `GET` | `/countries` | The markets that can be analysed. |
| `POST` | `/analyze` | Resolves the product, registers the analysis and starts the run. Answers `202` with the id. |
| `GET` | `/analyze/{id}` | The analysis as it stands (`Cache-Control: no-store`). |
| `GET` | `/analyze/{id}/stream` | Live status, passed through from the scoring service's SSE stream. |

Behind a reverse proxy, set `ROOT_PATH` (e.g. `/orchestrator`) so the Swagger
page builds its URLs under the right prefix.

## Quality controls

The narrative is written from a deterministic evidence packet, so a figure that
is not in the packet is not supported by anything. `app/quality.py` checks:

- `unsupported_numbers` — every number in the text, matched against the packet
  (exact, with a small tolerance for rounding);
- `language_violations` — phrases the method forbids ("proven demand",
  "guarantee");
- `tool_call_faults` — redundant and off-request tool calls;
- `judge_support` — a second model (`JUDGE_MODEL`, default `gpt-5.1`) reading
  the sentences that carry no figure, on `JUDGE_SAMPLE_RATE` (20 %) of the
  runs. Unset `JUDGE_API_KEY` skips it and says so in the log.

When the judge finds unsupported claims the narrative is sent back for a
**revision**, judged again, and kept only if it says less that is unsupported.

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
| `SCORING_INTERNAL_TOKEN` | — | Required by the scoring service's `/internal` routes. |
| `CHECKPOINTER` | `postgres` | `memory` only for tests. |
| `CHECKPOINTER_DB_HOST/PORT/NAME/USER/PASSWORD` | — | The service's own database. |
| `CHECKPOINTER_DATABASE_URL` | — | Wins over the parts when set. |
| `MEMORY_TTL_DAYS` | `30` | How long a "barren market" memory is trusted. |
| `JUDGE_API_KEY` | — | Unset skips the judged check. |
| `JUDGE_MODEL` | `gpt-5.1` | Deliberately not the model that wrote the text. |
| `JUDGE_SAMPLE_RATE` | `0.2` | |
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

- `docs/superpowers/specs/2026-09-23-agent-memory-design.md` — the store and
  what the agent remembers between runs.
- `docs/superpowers/specs/2026-08-04-marketmind-orchestrator-design.md` and
  `docs/superpowers/plans/2026-08-04-orchestrator-mvp.md` — the original
  design, kept as history; the banner at the top of each says what changed.
