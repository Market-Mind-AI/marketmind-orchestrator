# What the agent remembers between runs — design

**Date:** 2026-09-23 · **Status:** approved in chat, pending spec review
**Repo:** `marketmind-orchestrator` · **Branch:** `feat/agent-memory` (stacked on `feat/langfuse-llm-tracing`)

## Context

Piece 4 of the Langfuse program: (1) LLM tracing ✅, (2) prompt store ✅, (3) evaluations ✅, (4) long-term memory.

The agent already has **short-term memory**: the Postgres checkpointer records where each run got to, keyed by the analysis id, so a resumed run continues instead of restarting. It has no memory *between* runs.

Most of what a run learns is already persisted — by the scoring service, as domain data:

| Already kept | Where | Reused? |
|---|---|---|
| Search vocabulary per product and market | `product_catalog.localized_markets`, merged accumulate-never-replace | Yes: `enrich_node` says "only what is missing is generated, so a product analysed before costs no model call" |
| Reference-market candidates per product | `product_catalog.reference_candidates` | Yes, same merge |
| The buyer's corrections to the identified product | the confirmed fields are what get merged into the catalogue | Yes |

One thing nothing records: **which markets the sources simply do not hold data for**. Every analysis of such a market collects, fails, and retries the full window — the Mexico and Chile runs each spent about 70 seconds and three collection rounds rediscovering it.

## Goal

The agent keeps its own long-term memory, in a LangGraph store, of the markets it has found barren for a product, and uses it to stop repeating work it has already done.

## Non-goals

- Re-storing vocabulary or reference candidates. They live in the catalogue, and a second copy would be a second truth.
- Remembering which individual search terms produced real volume. That data is in the collectors, so it needs a scoring-service endpoint first. Worth doing later.
- Recording a correction *as a correction* (what the model said versus what the buyer changed it to). It needs identification and analysis, two separate requests, to be tied together.
- Changing anything the scoring service stores.

## Design

### The store

- `langgraph.store.postgres.AsyncPostgresStore`, opened from the same connection the checkpointer uses (`CHECKPOINTER_*`), in the lifespan, next to it. `setup()` creates its tables, as the checkpointer already does.
- `build_graph(checkpointer=None, store=None)` passes it to `compile(checkpointer=…, store=…)`. A node that declares `store: BaseStore` receives it; nodes that don't are unchanged.
- `CHECKPOINTER=memory` yields `InMemoryStore`, so the pair keeps behaving together.
- **Memory never fails a run.** Every store call is guarded: a failure is logged at debug level and the run proceeds exactly as it does today.

### What is remembered

One kind of memory, written and read as a whole:

- **Namespace:** `("markets", "barren")`
- **Key:** `f"{entity_key}:{country}"`
- **Value:** `{"recorded_at": <iso8601>, "analysis_id": <str>, "cause": <str>}`

### Writing

`give_up_node` already distinguishes why a run ended. When the cause is that the sources do not hold the data (not an outage, not the deadline), it records one entry per missing country. The other two causes are about the world being temporarily broken, and must not be remembered as facts about the product.

### Reading

A new step before collection checks each requested country against the memory:
- A record **fresher than `MEMORY_TTL_DAYS` (default 30)** means the market is declared unavailable immediately, without collecting or retrying.
- A **stale** record is ignored and deleted, so the market is tried again.
- The analysis record says why, naming the date: "no data when this was last checked, on <date>".
- The run still goes through `give_up`, so the reason reaches the buyer the same way as today.

### Forgetting

When a market scores, its record is deleted. A source that gains data therefore heals itself on the first successful run, and a wrong memory cannot outlive the fact.

### Visible in Langfuse

`recall_hits` is recorded as a score on every run: how many requested markets were answered from memory instead of collected. It is the measure that shows memory changing behaviour — 0 on a first run, and the number of barren markets on a repeat.

## Testing

With `InMemoryStore`, so no database is needed:
- A run that gives up because the sources hold no data writes one record per missing market.
- A run that gives up because a source was down, or because the deadline passed, writes nothing.
- A requested market with a fresh record is declared without collecting (the collection client is never called for it).
- A stale record is ignored, the market is collected, and the record is deleted.
- A market that scores has its record deleted.
- With no store configured, every path behaves exactly as today.
- `recall_hits` is recorded, and is 0 when nothing was recalled.

**Live check (paid, so ask first).** Analyse a barren market twice. The first run collects, retries and gives up, writing the memory. The second declares it in seconds, reports the date, and records `recall_hits=1`. About US$0.10, since identification and enrichment still run on the first analysis.

## Delivery

A `feat/agent-memory` branch and PR in `marketmind-orchestrator`, stacked on `feat/langfuse-llm-tracing`. The PR explains the split between what the catalogue already remembers and what the agent now remembers, and why the other three candidates were left out.

## Risks

- **A market wrongly remembered as barren** would hide a market the buyer asked for. Mitigated three ways: only the "sources hold no data" cause is remembered, the memory expires after 30 days, and any successful score deletes it.
- **Shared store, one namespace:** keys carry the entity and the country, so two products never collide.
- **Store growth:** one small row per barren (product, market). Nothing prunes it beyond the TTL check on read; at this scale that is fine.
