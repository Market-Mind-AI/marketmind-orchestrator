"""The steps of an analysis, in the order the scoring service's own loop ran them.

This is a faithful reimplementation, not an improvement. Each node calls a tool
and records what it learned; none of them computes anything. The MOS, the
indicators and the evidence stay where the method is versioned, and the only
model call is the narration, which explains an answer that is already decided.

Doing it faithfully first is what makes the change checkable: the same product,
the same countries and the same period have to produce the same MOS as the loop
this replaces. An improvement smuggled in here would make that comparison
meaningless, and there would be no second chance to make it.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from app import observability, quality
from app.scoring_client import ScoringClient, ScoringRejected, ScoringUnavailable

logger = logging.getLogger(__name__)

# How long to keep insisting on a source that is down. Past this the analysis
# fails with a reason rather than delivering part of a ranking: a country
# missing from a comparison is not a smaller answer, it is a different one.
INSIST_FOR = timedelta(hours=24)

# How many times to go back for countries that are still incomplete. A source
# that permanently lacks a datum would otherwise be asked forever, and the
# deadline alone would spend a day discovering it.
MAX_ATTEMPTS = 3


def _client() -> ScoringClient:
    return ScoringClient()


def _period_key(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return f"{now.year:04d}-{now.month:02d}"


async def start_node(state: dict) -> dict:
    """Open the run, and read back what was already decided.

    Reading rather than re-deciding is what makes resuming cheap. A resumed run
    must not pick a different product or a different set of countries than the
    one the buyer paid for.
    """
    analysis_id = state["analysis_id"]
    core = _client()
    stored = await core.read_analysis(analysis_id)
    await core.mark_running(analysis_id)

    return {
        "entity_key": stored.get("entity_key"),
        "canonical_title": stored.get("canonical_title"),
        "countries": list(stored.get("countries") or []),
        "user_prompt": stored.get("user_prompt"),
        # Decided here and only here, then carried in the checkpoint.
        "period_key": state.get("period_key") or _period_key(),
        "deadline_at": state.get("deadline_at") or (datetime.now(UTC) + INSIST_FOR).isoformat(),
        "attempts": state.get("attempts", 0),
    }


async def enrich_node(state: dict) -> dict:
    """Give the product a search vocabulary in each country.

    Only what is missing is generated, so a product analysed before costs no
    model call. Must run before collection, which reads these terms to know what
    to ask for.
    """
    entity_key = state.get("entity_key")
    if not entity_key:
        return {}

    core = _client()
    await core.record_progress(state["analysis_id"], "preparing the product's vocabulary")
    await core.enrich_terms(entity_key, state["countries"], with_candidates=True)

    # The category behind a branded product is enriched WITHOUT candidates: the
    # demand ruler belongs to the brand, and the category is the scale the
    # brand's demand is read against — context, never scored.
    family = state.get("generic_entity_key")
    if family:
        await core.enrich_terms(family, state["countries"], with_candidates=False)
    return {}


async def collect_node(state: dict) -> dict:
    """Collect what each country is missing, and keep the tasks.

    The tasks are what the next decision is made on. A country that comes back
    incomplete is either waiting on a source that is down -- insist -- or
    reading a source that does not hold the datum -- declare it and move on.
    """
    core = _client()
    analysis_id = state["analysis_id"]
    entity_key = state.get("entity_key")

    tasks: list[dict[str, Any]] = []
    collected: list[str] = []
    for country in state["countries"]:
        await core.record_progress(analysis_id, f"collecting {country}")
        try:
            answer = await core.ensure_coverage(country, entity_key)
            # The words this country was actually searched for. Logged because
            # they are the one thing a model chose that nothing else exposes:
            # get the native term wrong and the analysis is immaculate and
            # about the wrong product.
            logger.info(
                "SEARCH TERMS %s (%s, %s): product=%s | category=%s",
                country,
                answer.get("location"),
                answer.get("language"),
                answer.get("search_terms"),
                answer.get("category_terms"),
            )
            tasks.extend(answer.get("tasks", []))
            collected.append(country)
        except ScoringUnavailable as exc:
            # Transient by definition. Recorded, not raised: one country's
            # source being down must not abandon the others' collection.
            logger.warning("Collection for %s could not run: %s", country, exc)
            tasks.append({"country": country, "status": "failed", "error_message": str(exc)})

    return {"coverage_tasks": tasks, "countries_collected": collected}


async def ruler_node(state: dict) -> dict:
    """Settle the ruler the countries' demand is measured against.

    The ceiling is the largest real search volume among the product's reference
    markets, so those are collected regardless of what was asked about. That is
    what keeps the ruler from moving with the query -- and when the period's
    champion is already known, the collection is skipped entirely, because it is
    the slowest work in the system and it produces the same answer.
    """
    entity_key = state.get("entity_key")
    if not entity_key:
        return {"demand_ceiling": None, "demand_ruler": None, "champion_cached": False}

    core = _client()
    await core.record_progress(state["analysis_id"], "settling the demand ruler")
    known = await core.candidates_coverage(entity_key, state["countries"], state["period_key"])
    ceiling, ruler = await core.demand_ceiling(entity_key, state["countries"], state["period_key"])
    return {
        "champion_cached": bool(known.get("champion_cached")),
        "cached_champion_country": known.get("champion_country"),
        "cached_champion_mos": known.get("champion_mos"),
        "demand_ceiling": ceiling,
        "demand_ruler": ruler,
    }


async def measure_node(state: dict) -> dict:
    """Compute each country's indicators, then make sure the period has weights.

    The weights are deterministic and cheap, and calling this is not optional: a
    period with none excludes every country from the MOS with the reason "no
    calibrated weight", which reads like missing data and is not.
    """
    core = _client()
    analysis_id = state["analysis_id"]
    for country in state["countries"]:
        await core.record_progress(analysis_id, f"measuring {country}")
        await core.compute_indicators(
            state.get("entity_key"), country, state["period_key"], state.get("demand_ceiling")
        )
    await core.calibrate_weights(state["period_key"])
    return {}


async def score_node(state: dict) -> dict:
    """Compute the MOS. No arithmetic happens here; it happens where it is versioned."""
    entity_key = state.get("entity_key")
    if not entity_key:
        return {"mos_results": []}

    core = _client()
    await core.record_progress(state["analysis_id"], "scoring the markets")
    results = await core.compute_mos(entity_key, state["countries"], state["period_key"])
    return {"mos_results": results, "attempts": state.get("attempts", 0) + 1}


def incomplete_countries(state: dict) -> list[str]:
    """Requested countries with no usable score.

    Scored is the only thing that counts as done. A row that exists but carries
    no MOS is a country that was measured and could not be ranked, which is the
    same gap as one that was never measured.
    """
    requested = set(state.get("countries") or [])
    scored = {
        row.get("country")
        for row in (state.get("mos_results") or [])
        if row.get("mos") is not None and row.get("country") in requested
    }
    return sorted(requested - scored)


def _has_transient_failure(state: dict, countries: list[str]) -> bool:
    """Whether any incomplete country is waiting on a source that failed.

    This is the distinction the whole retry rule rests on. A collection task
    that FAILED is a source that is down, and it comes back. A task that
    finished without failing and still left the indicator missing is a source
    that does not hold that datum, and no amount of asking will produce it.
    """
    wanted = set(countries)
    return any(
        task.get("status") == "failed" and (task.get("country") in wanted or task.get("country") is None)
        for task in (state.get("coverage_tasks") or [])
    )


def _past_deadline(state: dict) -> bool:
    deadline = state.get("deadline_at")
    if not deadline:
        return False
    return datetime.now(UTC) >= datetime.fromisoformat(deadline)


def after_scoring(state: dict) -> str:
    """Insist, give up, or carry on -- the decision the analysis turns on.

    Nothing partial goes out. A country missing from a ranking is not a smaller
    answer, it is a different one, and the buyer paid for the one they asked
    for. So an incomplete run goes back to collect, for as long as the cause is
    a source that is down.

    It stops going back when the cause is permanent -- the source does not have
    the datum, and the country is declared rather than waited on -- when the
    attempts run out, or when the deadline passes. Those end the run with a
    reason instead of delivering half a comparison.
    """
    missing = incomplete_countries(state)
    if not missing:
        return "complete"
    if _past_deadline(state):
        return "give_up"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "give_up"
    if not _has_transient_failure(state, missing):
        # Permanent. Asking again produces the same nothing.
        return "give_up"
    return "insist"


async def champion_node(state: dict) -> dict:
    """Surface the product's best market, so the request can be benchmarked.

    The MOS uses fixed anchors, so scoring markets nobody asked about never
    changes what a requested one scores. When the best of them beats every
    requested market, it is worth showing; when a requested market beats the
    stored champion, the stored one is replaced, because leaving a beaten
    champion in place keeps it wrong for every analysis after.
    """
    entity_key = state.get("entity_key")
    results = state.get("mos_results") or []
    if not entity_key or not results:
        return {}

    core = _client()
    await core.record_progress(state["analysis_id"], "looking for the product's best market")
    scored = [r for r in results if r.get("mos") is not None]
    best_requested = max(scored, key=lambda r: float(r["mos"]), default=None)

    if state.get("champion_cached"):
        # The crown means "beats everything you asked about". A stored champion
        # that a requested market now beats is not one, and saying otherwise
        # produced a visibly false claim once: a champion stored at 66.37 stayed
        # crowned in a later analysis where a requested market scored 69.
        cached_country = state.get("cached_champion_country")
        cached_mos = state.get("cached_champion_mos")
        still_wins = (
            cached_country is not None
            and cached_mos is not None
            and (best_requested is None or float(cached_mos) > float(best_requested["mos"]))
        )
        if still_wins:
            return {"champion_country": cached_country}

        if best_requested is None:
            return {}

        # Promoted, not merely displaced. The stored value means "this product's
        # best known market", so leaving a beaten one in place keeps it wrong
        # for every analysis after this one — a later request for two other
        # markets would reload it and crown it again. Cheap: the numbers are
        # already in hand and no candidate collection is involved.
        await core.set_champion(
            entity_key,
            state["period_key"],
            country=best_requested["country"],
            mos=float(best_requested["mos"]),
        )
        return {"champion_country": best_requested["country"]}

    candidates = await core.score_candidates(
        entity_key, state["countries"], state["period_key"], state.get("demand_ceiling")
    )
    everything = [*scored, *[c for c in candidates if c.get("mos") is not None]]
    champion = max(everything, key=lambda r: float(r["mos"]), default=None)

    # Recorded even when nothing could be scored: the period is what marks the
    # work as done, so "nothing scorable this month" has to be stored or the
    # same fruitless collection runs on every later analysis.
    await core.set_champion(
        entity_key,
        state["period_key"],
        country=champion["country"] if champion else None,
        mos=float(champion["mos"]) if champion else None,
    )

    requested = set(state["countries"])
    surfaced = list(results)
    if champion and champion["country"] not in requested:
        surfaced = [*results, champion]
    return {"champion_country": champion["country"] if champion else None, "mos_results": surfaced}


async def evidence_node(state: dict) -> dict:
    """Assemble what the narrative is written from, and what the frontend renders.

    Deterministic and free: it reads stored results and collects nothing. No
    model is involved, which is what makes the narrative checkable afterwards --
    a claim that is not in this packet is not supported by anything.
    """
    entity_key = state.get("entity_key")
    if not entity_key:
        return {"breakdown": None}

    core = _client()
    await core.record_progress(state["analysis_id"], "assembling the evidence")
    packet = await core.build_evidence(
        entity_key, state["countries"], state["period_key"], state.get("demand_ruler")
    )
    if packet is not None:
        requested = set(state["countries"])
        for entry in packet.get("countries") or []:
            entry["requested"] = entry.get("country") in requested
            entry["is_champion"] = entry.get("country") == state.get("champion_country")
        packet["champion_country"] = state.get("champion_country")
        packet["requested_countries"] = list(state["countries"])
    return {"breakdown": packet}


async def deliver_node(state: dict) -> dict:
    """Record the answer, without waiting for the prose.

    The scores are the answer; the narrative is a reading of them. One model
    call once took longer than the entire rest of the analysis, so the request
    completes with its numbers and the text is attached a moment later.
    """
    results = state.get("mos_results") or []
    requested = set(state["countries"])
    for row in results:
        row["requested"] = row.get("country") in requested
        row["is_champion"] = row.get("country") == state.get("champion_country")

    await _client().complete_analysis(
        state["analysis_id"], result=results, breakdown=state.get("breakdown")
    )

    requested_scored = len([r for r in results if r.get("mos") is not None and r.get("country") in requested])
    observability.record(
        "analysis.complete",
        True,
        comment=f"{requested_scored} of {len(requested)} requested markets scored",
    )
    observability.record("analysis.attempts", float(state.get("attempts", 1)))
    faults = quality.tool_call_faults(state.get("coverage_tasks") or [], list(requested))
    # Split, because "accuracy" as one number says nothing about what to fix.
    # A redundant collection is where cost leaks; an off-request one is a call
    # that should never have been made.
    observability.record("tools.redundant_collections", float(faults["redundant"]))
    observability.record("tools.off_request_collections", float(faults["off_request"]))
    return {}


async def narrate_node(state: dict) -> dict:
    """Turn the packet into prose, and never let that undo a delivered answer.

    Everything here is swallowed, not just the model call. An earlier version
    guarded only the generation, and a failure in the write marked an
    already-successful analysis as failed.
    """
    packet = state.get("breakdown")
    if not packet:
        return {}

    try:
        core = _client()
        # Said before the call, not after. The request is already complete at
        # this point, so without it the reader sees a finished analysis with an
        # explanation that is simply absent, and no way to tell "coming" from
        # "never".
        await core.record_progress(state["analysis_id"], "writing the explanation")
        narrative = await core.narrate(
            packet, state.get("canonical_title") or "", state.get("user_prompt")
        )
        await core.attach_narrative(state["analysis_id"], narrative)
        await core.record_progress(state["analysis_id"], None)
        await _check_narrative(narrative, packet)
        return {"narrative": narrative}
    except (ScoringUnavailable, ScoringRejected) as exc:
        logger.exception("Analysis %s is delivered, but attaching its narrative failed", state["analysis_id"])
        return {"errors": [f"narration failed: {exc}"]}


async def give_up_node(state: dict) -> dict:
    """End the run with a reason, rather than delivering part of a comparison.

    Which countries, and why each one. A failure nobody can read is the one the
    buyer asks about, and "it failed" does not say whether to try again in an
    hour or never.
    """
    missing = incomplete_countries(state)
    if _past_deadline(state):
        cause = "the sources stayed unavailable for the full retry window"
    elif _has_transient_failure(state, missing):
        cause = f"collection kept failing after {state.get('attempts', 0)} attempts"
    else:
        cause = "the sources do not hold the data for these markets"

    reason = f"Analysis incomplete for {', '.join(missing) or 'every requested market'}: {cause}."
    await _client().fail_analysis(state["analysis_id"], reason)
    return {"errors": [reason]}



async def _check_narrative(narrative: str | None, packet: dict[str, Any]) -> None:
    """What the text claims, measured against what the evidence holds.

    Possible only because no number in the answer comes from a model: the
    narrative is written from a deterministic packet, so a figure that is not
    in the packet is not supported by anything. That check is exact and free,
    so it runs on every analysis.

    The judge is for what it cannot reach — sentences that claim something
    without stating a figure — and is sampled, because a second model on every
    run doubles the cost of the thing being measured.

    Nothing here can fail the analysis. It is already delivered.
    """
    if not narrative:
        observability.record("narrative.written", False, comment="the model returned nothing")
        return

    try:
        unsupported = quality.unsupported_numbers(narrative, packet)
        observability.record(
            "narrative.unsupported_figures",
            float(len(unsupported)),
            comment=("none" if not unsupported else f"stated but not in the evidence: {unsupported}"),
        )

        violations = quality.language_violations(narrative)
        observability.record(
            "narrative.language_violations",
            float(len(violations)),
            comment=("none" if not violations else "; ".join(violations)),
        )

        from app.config import get_settings

        import random

        if random.random() > get_settings().judge_sample_rate:
            logger.info("QUALITY narrative.judged=skipped (outside the judged sample)")
            return

        verdict = await quality.judge_support(narrative, packet)
        if verdict is None:
            # Already said why, in quality.judge_support. No score is recorded:
            # an absent judgement must not be readable as a clean one.
            return
        claims = verdict.get("unsupported_claims") or []
        observability.record(
            "narrative.unsupported_claims",
            float(len(claims)),
            comment=("none" if not claims else "; ".join(str(c) for c in claims)[:500]),
        )
    except Exception:  # noqa: BLE001 — a measurement must never break what it measures
        logger.exception("Could not check the narrative; the analysis is unaffected")
