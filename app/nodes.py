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

from app import metrics, observability, quality
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
    _note_catalogo(stored)

    return {
        "entity_key": stored.get("entity_key"),
        "canonical_title": stored.get("canonical_title"),
        "countries": list(stored.get("countries") or []),
        "user_prompt": stored.get("user_prompt"),
        # Decided here and only here, then carried in the checkpoint.
        "period_key": state.get("period_key") or _period_key(),
        "started_at": state.get("started_at") or datetime.now(UTC).isoformat(),
        "deadline_at": state.get("deadline_at") or (datetime.now(UTC) + INSIST_FOR).isoformat(),
        "attempts": state.get("attempts", 0),
    }



# Como o catálogo chegou na entidade, e de que TIPO foi a coincidência. Os três
# primeiros são reaproveitamento e só o quarto é produto inédito, mas a
# diferença entre os três importa: `alias` é igualdade, `trigram` é semelhança
# de ESCRITA com número próprio, e `llm` é coincidência de SIGNIFICADO, que não
# tem número nenhum — `quadcopter` e `drone` não se parecem em nada escrito.
#
# Foi por isso que o passo de vetor saiu: `product_catalog.embedding`
# (vector(1536)) existiu até 14/08/2026 e foi medido contra os 89 termos reais
# do catálogo — o trigram resolve 71 deles com precisão de 100%, e nos 18 que
# sobram nenhuma distância de cosseno decide, porque a relação é conhecimento e
# não proximidade de string. Ver a migration 027 do serviço de scoring.
# O limiar do trigram no serviço de scoring (`catalog_trigram_similarity_threshold`).
# Não é lido de lá porque este processo não tem essa configuração; é repetido
# aqui de propósito e só para ROTULAR, nunca para decidir — quem decide é quem
# tem o número.
_LIMIAR_TRIGRAM = 0.4
# Quão perto do limiar ainda é "quase não passou". Uma coincidência de 0,41
# contra um corte de 0,40 é decisão de identidade de produto tirada por um fio,
# e antes de 23/09/2026 nada mostrava isso. Medido no mesmo dia: "cafetera
# italiana para espresso" casou com "cafeteira italiana moka" em 41,0%.
_MARGEM_LIMIAR = 0.05

_COINCIDENCIA = {
    "alias": ("exact", "an exact alias already pointed at it"),
    "trigram": ("spelling", "its name is spelled almost the same"),
    "llm": ("meaning", "it means the same product, by the model's judgement"),
    "created": ("none", "nothing in the catalogue was this product"),
}


def _note_catalogo(stored: dict[str, Any]) -> None:
    """Say whether this analysis is about a product the catalogue already knew,
    which kind of match found it, and how close it was.

    It is the difference between a catalogue that works and one that quietly
    duplicates, and nothing else in the run reveals it: a fresh entity and a
    matched one produce identically shaped analyses.

    Read back from the record rather than observed at resolve time, because the
    resolve happens in POST /analyze — before the graph runs, so before there
    is a trace to attach anything to — and because a resumed run does not
    resolve the product again.
    """
    metodo = stored.get("catalog_method")
    titulo = stored.get("canonical_title")
    if not metodo:
        # Análises registradas antes de 23/09/2026 não têm o campo. Dizer isso
        # é melhor que inventar um método ou omitir o fato.
        observability.note(
            "catalog_resolution",
            summary=f"{titulo!r}: how it was matched was not recorded (analysis predates the field)",
            detail={"entity_key": stored.get("entity_key"), "canonical_title": titulo},
        )
        return

    reaproveitou = metodo != "created"
    tipo, explicacao = _COINCIDENCIA.get(metodo, ("unknown", f"resolved by {metodo}"))
    score = stored.get("catalog_match_score")
    motivo = stored.get("catalog_match_reasoning")

    # O par: o que foi perguntado e em que foi parar. Com um lado só, "picanha,
    # coincidência exata" não diz nada a quem pediu "top sirloin cap", e uma
    # coincidência de 41% não pode ser julgada por ninguém.
    perguntado = stored.get("catalog_query_text")
    veredito = "REUSED the catalogue entry" if reaproveitou else "CREATED a new catalogue entry"
    alvo = f"{perguntado!r} -> {titulo!r}" if perguntado and perguntado != titulo else f"{titulo!r}"
    resumo = f"{alvo}: {veredito} — match by {tipo}, {explicacao}"
    no_fio = False
    if score is not None:
        # Em porcentagem porque é assim que se lê "quanto coincidiu". O número
        # cru (0-1) fica no metadata para quem for comparar execuções.
        resumo += f" ({float(score) * 100:.1f}% similar)"
        no_fio = float(score) < _LIMIAR_TRIGRAM + _MARGEM_LIMIAR
        if no_fio:
            resumo += f" — BORDERLINE, barely over the {_LIMIAR_TRIGRAM * 100:.0f}% threshold"
    if motivo:
        resumo += f' — "{motivo}"'

    observability.note(
        "catalog_resolution",
        summary=resumo,
        detail={
            # A flag que responde "aproveitou ou criou", sozinha e sem precisar
            # interpretar o método.
            "reused_existing_entry": reaproveitou,
            "match_type": tipo,
            "method": metodo,
            "match_percent": None if score is None else round(float(score) * 100, 1),
            # Vale por si: uma identidade decidida a um ponto do corte merece
            # ser achável sem ninguém ter que comparar com o limiar na mão.
            "borderline": no_fio,
            "match_score": None if score is None else float(score),
            "match_reasoning": motivo,
            "queried_text": perguntado,
            "entity_key": stored.get("entity_key"),
            "canonical_title": titulo,
        },
    )


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
    reaproveitado: dict[str, int] = {}
    sem_cobertura: dict[str, list[str]] = {}
    relacionadas_frescas = 0
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
            # O outro lado da conta, que as tarefas não contam: o que já estava
            # no banco nem vira tarefa. Sem isto uma fonte inteiramente
            # aproveitada some do relato.
            for fonte, quantos in (answer.get("reused") or {}).items():
                reaproveitado[fonte] = reaproveitado.get(fonte, 0) + quantos
            for fonte in answer.get("not_covering") or []:
                sem_cobertura.setdefault(fonte, []).append(country)
            if answer.get("related_fresh"):
                relacionadas_frescas += 1
            collected.append(country)
        except ScoringUnavailable as exc:
            # Transient by definition. Recorded, not raised: one country's
            # source being down must not abandon the others' collection.
            logger.warning("Collection for %s could not run: %s", country, exc)
            tasks.append({"country": country, "status": "failed", "error_message": str(exc)})

    _note_fontes(
        tasks, reused=reaproveitado, not_covering=sem_cobertura, related_fresh=relacionadas_frescas
    )
    return {"coverage_tasks": tasks, "countries_collected": collected}


# Os nomes das fontes como uma pessoa as chama, não como o enum as escreve.
_FONTES = {
    "worldbank": "World Bank",
    "oecd": "OECD",
    "undata": "UN Data",
    "pytrends": "Google Ads (via DataForSEO)",
}


def _note_fontes(
    tasks: list[dict[str, Any]],
    *,
    reused: dict[str, int] | None = None,
    not_covering: dict[str, list[str]] | None = None,
    related_fresh: int = 0,
    nome: str = "data_sources",
) -> None:
    """Say, for EACH of the four sources, what was downloaded and what was
    already here.

    Each of the four is named every time, even when it did nothing, because a
    source that is silently absent from the line is indistinguishable from one
    that was never consulted — which is the very question the line exists to
    answer. The first version only counted tasks, and a source whose data was
    entirely fresh produces no task at all: the first real run said only
    "Google Ads: 2 fetched" and said nothing whatsoever about the World Bank,
    the OECD or UN Data, which had supplied every macro indicator from the
    database.

    Three different silences, told apart:
      reused        the datum was here, inside its freshness window, and
                    nothing was asked of the source. This is the good case and
                    the expensive one to confuse with the others.
      not covering  the source does not have this country at all — read from
                    the country's own mapping, not from a hand-kept list.
      nothing       no indicator of this analysis comes from that source.
    """
    reused = reused or {}
    not_covering = not_covering or {}

    por_fonte: dict[str, dict[str, int]] = {}
    sem_fonte = 0
    for task in tasks:
        fonte = task.get("source")
        if not fonte:
            # `collect_node` fabrica isto quando o scoring nem respondeu: não há
            # fonte porque não se chegou a nenhuma.
            sem_fonte += 1
            continue
        contagem = por_fonte.setdefault(fonte, {"fetched": 0, "failed": 0})
        if task.get("status") == "failed":
            contagem["failed"] += 1
        else:
            contagem["fetched"] += 1

    partes = []
    detalhe: dict[str, Any] = {}
    for fonte, rotulo in _FONTES.items():
        c = por_fonte.get(fonte, {"fetched": 0, "failed": 0})
        aproveitados = reused.get(fonte, 0)
        paises_sem = not_covering.get(fonte, [])
        detalhe[fonte] = {"fetched": c["fetched"], "reused": aproveitados, "failed": c["failed"]}

        if c["fetched"] or aproveitados or c["failed"]:
            pedaco = f"{rotulo}: {c['fetched']} fetched, {aproveitados} reused from the database"
            if c["failed"]:
                pedaco += f", {c['failed']} failed"
            if fonte == "pytrends" and related_fresh:
                # Contado à parte: é UM conjunto de sementes por país, não N
                # termos. Somar os dois numa contagem só misturaria unidades.
                pedaco += f" (+ related keywords already stored for {related_fresh} country/ies)"
                detalhe[fonte]["related_reused_countries"] = related_fresh
        elif paises_sem:
            pedaco = f"{rotulo}: does not cover {', '.join(paises_sem)}"
            detalhe[fonte]["not_covering"] = paises_sem
        elif fonte == "pytrends" and related_fresh:
            pedaco = f"{rotulo}: nothing fetched — every search term and related keyword was already stored"
            detalhe[fonte]["related_reused_countries"] = related_fresh
        else:
            pedaco = f"{rotulo}: nothing needed from it"
        partes.append(pedaco)

    if sem_fonte:
        partes.append(f"{sem_fonte} collection(s) could not even run — the scoring service did not answer")
        detalhe["unreachable"] = sem_fonte

    total_buscado = sum(c["fetched"] for c in por_fonte.values())
    total_aproveitado = sum(reused.values())
    observability.note(nome, summary=" | ".join(partes), detail=detalhe)
    # Também como número, que é o que dá para comparar entre execuções: uma
    # análise que de repente baixa tudo de novo aparece como uma subida aqui,
    # sem ninguém precisar abrir o trace.
    observability.record(
        f"{nome}.fetched",
        total_buscado,
        comment=f"downloaded from a source; {total_aproveitado} more were already in the database",
    )


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
    # O passo mais caro do sistema, e até 23/09/2026 o único sem relato nenhum:
    # a coleta dos mercados de referência não devolvia nada, então nem o custo
    # nem o reaproveitamento dela apareciam em lugar algum. Fato próprio, e não
    # somado ao da coleta dos países pedidos, porque é outra decisão: aqui se
    # coleta país que NINGUÉM pediu, para a régua não mudar com a pergunta.
    if known.get("champion_cached"):
        observability.note(
            "data_sources_reference_markets",
            summary=(
                "skipped entirely: this period's champion is already known, and "
                "re-collecting the reference markets produces the same answer"
            ),
            detail={"champion_cached": True, "champion_country": known.get("champion_country")},
        )
    else:
        _note_fontes(
            known.get("tasks") or [],
            reused=known.get("reused") or {},
            related_fresh=int(known.get("related_fresh") or 0),
            nome="data_sources_reference_markets",
        )
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

    # Measured here, where the buyer actually has the answer — not after the
    # narrative, which is attached a moment later and is not what was waited for.
    metrics.task_completion_rate(results, requested)
    metrics.response_latency_seconds(state.get("started_at"))
    metrics.token_consumption()
    metrics.tool_invocation_accuracy(state.get("coverage_tasks") or [], list(requested))
    observability.record("analysis.attempts", float(state.get("attempts", 1)))
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
        await _check_narrative(narrative, packet, state["analysis_id"])
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

    # Reported for a failed run too. A completion rate measured only on the runs
    # that completed is always 1.0, and a latency measured only on those hides
    # exactly the runs that took longest.
    requested = set(state["countries"])
    metrics.task_completion_rate(state.get("mos_results") or [], requested)
    metrics.response_latency_seconds(state.get("started_at"), what="given up")
    metrics.token_consumption()
    metrics.tool_invocation_accuracy(state.get("coverage_tasks") or [], list(requested))
    observability.record("analysis.attempts", float(state.get("attempts", 1)))
    return {"errors": [reason]}



async def _revise_narrative(
    analysis_id: str, narrative: str, claims: list[str], packet: dict[str, Any]
) -> None:
    """Ask for the sentences to be rewritten, and only keep the rewrite if it
    is actually better.

    Measuring this was not enough. A run today had every figure supported
    (`hallucination_rate=0.0`, 0 of 134) and still asserted that "the USA is
    not part of the ranking" and that "Spain is the more coherent export bet" —
    the second of which the data cannot support at all, covering no cost, no
    tariff and no sanitary requirement. The project's invariant is that the
    model never computes a number, and that is exactly what makes the numeric
    check exact; it was never a defence against inventing the CONCLUSION.

    The revision is judged by the same judge, and kept only if it says less
    that is unsupported. A rewrite that makes it worse is discarded — a second
    model call cannot be allowed to damage an answer already delivered.
    """
    revista = await _client().revise_narrative(narrative, claims)
    if not revista:
        observability.record("narrative.revised", False, comment="the revision did not come back")
        return

    novo_veredito = await quality.judge_support(revista, packet)
    if novo_veredito is None:
        # Sem segunda opinião não há como afirmar que melhorou, e trocar às
        # cegas é exatamente o risco que esta função existe para não correr.
        observability.record("narrative.revised", False, comment="the revision could not be judged")
        return

    restantes = [str(c) for c in (novo_veredito.get("unsupported_claims") or [])]
    if len(restantes) >= len(claims):
        observability.record(
            "narrative.revised",
            False,
            comment=f"discarded: {len(restantes)} unsupported claims against the original's {len(claims)}",
        )
        return

    await _client().attach_narrative(analysis_id, revista)
    observability.record(
        "narrative.revised",
        True,
        comment=f"unsupported claims {len(claims)} -> {len(restantes)}",
    )
    observability.record(
        "narrative.unsupported_claims_final",
        float(len(restantes)),
        comment=("none" if not restantes else "; ".join(restantes)[:500]),
    )


async def _check_narrative(narrative: str | None, packet: dict[str, Any], analysis_id: str = "") -> None:
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
        metrics.hallucination_rate(narrative, packet, unsupported)

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
        claims = [str(c) for c in (verdict.get("unsupported_claims") or [])]
        observability.record(
            "narrative.unsupported_claims",
            float(len(claims)),
            comment=("none" if not claims else "; ".join(claims)[:500]),
        )
        if claims:
            await _revise_narrative(analysis_id, narrative, claims, packet)
    except Exception:  # noqa: BLE001 — a measurement must never break what it measures
        logger.exception("Could not check the narrative; the analysis is unaffected")
