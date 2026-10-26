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
from app.config import get_settings
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

# Said in give_up_node when the sources were just found not to hold a market's
# data. One string, so the reason on disk and the reason a test looks for do not
# drift apart.
_NO_DATA_CAUSE = "the sources do not hold the data for these markets"


def _client() -> ScoringClient:
    return ScoringClient()


def _period_key(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return f"{now.year:04d}-{now.month:02d}"


def _collection_timeout(state: dict) -> float | None:
    """How long a collecting call may take: what is left until the deadline.

    Waiting past the deadline buys nothing — the run gives up there anyway —
    and waiting less turns the scoring service's own retries into a timeout
    here. Never below the floor, so a call made near the deadline still lets
    the gap in flight finish. None outside a run with a deadline, which leaves
    the client's own ceiling.
    """
    deadline = state.get("deadline_at")
    if not deadline:
        return None
    remaining = (datetime.fromisoformat(deadline) - datetime.now(UTC)).total_seconds()
    return max(remaining, get_settings().scoring_collection_timeout_floor_seconds)


async def start_node(state: dict) -> dict:
    """Open the run, and read back what was already decided.

    Reading rather than re-deciding is what makes resuming cheap. A resumed run
    must not pick a different product or a different set of countries than the
    one the buyer paid for.
    """
    analysis_id = state["analysis_id"]
    core = _client()
    stored = await core.read_analysis(analysis_id)
    # Decided here and only here, then carried in the checkpoint — and told to
    # the scoring service, which stores the first one it hears and is what
    # closes an analysis left stuck past it. A resumed run sends the same one.
    deadline_at = state.get("deadline_at") or (datetime.now(UTC) + INSIST_FOR).isoformat()
    await core.mark_running(analysis_id, deadline_at=deadline_at)
    _note_catalogo(stored)

    return {
        "entity_key": stored.get("entity_key"),
        "canonical_title": stored.get("canonical_title"),
        "brand_key": stored.get("brand_key"),
        "brand_name": stored.get("brand_name"),
        "countries": list(stored.get("countries") or []),
        "user_prompt": stored.get("user_prompt"),
        "language": stored.get("language"),
        # Decided here and only here, then carried in the checkpoint.
        "period_key": state.get("period_key") or _period_key(),
        "started_at": state.get("started_at") or datetime.now(UTC).isoformat(),
        "deadline_at": deadline_at,
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
    # Quem diz é o serviço que tem o limiar. Este processo chegou a repetir o
    # 0.4 só para rotular, e número repetido envelhece sozinho: mudar o limiar
    # lá deixaria o rótulo daqui mentindo sem nada acusar. Uma coincidência de
    # 41% contra um corte de 40% é identidade de produto decidida por um fio.
    no_fio = bool(stored.get("catalog_match_borderline"))
    if score is not None:
        # Em porcentagem porque é assim que se lê "quanto coincidiu". O número
        # cru (0-1) fica no metadata para quem for comparar execuções.
        resumo += f" ({float(score) * 100:.1f}% similar)"
        if no_fio:
            resumo += " — BORDERLINE, barely over the catalogue's similarity threshold"
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
    marca = state.get("brand_key")
    await core.enrich_terms(entity_key, state["countries"], with_candidates=True, brand_key=marca)

    # A CATEGORIA do mesmo produto, quando a análise é de marca: desde
    # 24/09/2026 não há entidade-família, é o outro vocabulário da MESMA
    # entidade (sem marca). Sem candidatos, porque a régua de demanda é da
    # marca e a categoria é a escala contra a qual ela se lê — contexto, nunca
    # pontuado.
    if marca:
        await core.enrich_terms(entity_key, state["countries"], with_candidates=False, brand_key=None)
    return {}


async def collect_node(state: dict) -> dict:
    """Collect what each country is missing, and keep the tasks.

    The tasks are what the next decision is made on. A country that comes back
    incomplete is either waiting on a source that is down -- insist -- or
    reading a source that does not hold the datum -- declare it and move on.

    The scoring service is asked first which markets had no demand recently: a
    market whose every search term was answered with nothing inside the
    demand's freshness window is declared without spending the collection
    round -- the slowest part of this, sometimes several retries against a
    source -- that would only rediscover the same nothing. That answer is read
    from the data, per product AND brand, and heals itself when a source gains
    volume; the memory this replaces was keyed by product alone and trusted for
    a window of its own.
    """
    core = _client()
    analysis_id = state["analysis_id"]
    entity_key = state.get("entity_key")

    tasks: list[dict[str, Any]] = []
    collected: list[str] = []
    reaproveitado: dict[str, int] = {}
    sem_cobertura: dict[str, list[str]] = {}
    relacionadas_frescas = 0
    absent = await _absent_markets(core, state)
    for country in state["countries"]:
        if country in absent:
            continue

        await core.record_progress(analysis_id, f"collecting {country}")
        try:
            answer = await core.ensure_coverage(
                country,
                entity_key,
                state["period_key"],
                state.get("brand_key"),
                timeout=_collection_timeout(state),
            )
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
            observability.flag("WARNING", f"collection for {country} could not run: {exc}")
            tasks.append({"country": country, "status": "failed", "error_message": str(exc)})

    _note_fontes(
        tasks, reused=reaproveitado, not_covering=sem_cobertura, related_fresh=relacionadas_frescas
    )
    return {"coverage_tasks": tasks, "countries_collected": collected, "absent_countries": absent}


async def _absent_markets(core: ScoringClient, state: dict) -> dict[str, str]:
    """The requested markets with no recent demand, mapped to when that was
    last checked. Best effort: an answer that does not come means collecting
    everything, which is what a run did before it could ask."""
    entity_key = state.get("entity_key")
    if not entity_key:
        return {}
    try:
        answer = await core.demand_absence(
            entity_key, state["countries"], state["period_key"], state.get("brand_key")
        )
    except (ScoringUnavailable, ScoringRejected) as exc:
        logger.warning("Could not ask which markets lack demand; collecting all of them: %s", exc)
        return {}
    return {
        market["country"]: market.get("checked_at") or ""
        for market in answer.get("markets") or []
        if market.get("absent") and market.get("country") in set(state["countries"])
    }


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
    champion is already known against the current ruler, the scoring service
    skips the collection, because it is the slowest work in the system and it
    produces the same answer. Who the champion is, is not read here: that is
    /champion/crown's answer, after the countries are scored.
    """
    entity_key = state.get("entity_key")
    if not entity_key:
        return {"demand_ceiling": None, "demand_ruler": None}

    core = _client()
    await core.record_progress(state["analysis_id"], "settling the demand ruler")
    known = await core.candidates_coverage(
        entity_key,
        state["countries"],
        state["period_key"],
        state.get("brand_key"),
        timeout=_collection_timeout(state),
    )
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
            detail={"champion_cached": True},
        )
    else:
        _note_fontes(
            known.get("tasks") or [],
            reused=known.get("reused") or {},
            related_fresh=int(known.get("related_fresh") or 0),
            nome="data_sources_reference_markets",
        )
    ceiling, ruler = await core.demand_ceiling(
        entity_key, state["countries"], state["period_key"], state.get("brand_key")
    )
    return {"demand_ceiling": ceiling, "demand_ruler": ruler}


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
            state.get("entity_key"),
            country,
            state["period_key"],
            state.get("demand_ceiling"),
            state.get("brand_key"),
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
    results = await core.compute_mos(
        entity_key, state["countries"], state["period_key"], state.get("brand_key")
    )
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
    """Ask the scoring service for the product's best market, so the request
    can be benchmarked.

    Crowning is its decision, not this one's: it holds the demand ruler, and a
    stored champion only counts when it was measured against the current one.
    Deciding it here read the stored champion in one call and crowned it after
    the ruler had moved in the next. What comes back is used as it is — when
    the champion was not asked about, its row joins the delivered ones, because
    the crown is drawn from the rows the buyer receives.
    """
    entity_key = state.get("entity_key")
    results = state.get("mos_results") or []
    if not entity_key or not results:
        return {}

    core = _client()
    await core.record_progress(state["analysis_id"], "looking for the product's best market")
    crowned = await core.crown_champion(
        entity_key,
        state["period_key"],
        state["countries"],
        results,
        state.get("brand_key"),
        timeout=_collection_timeout(state),
    )

    # Only the requested rows are this run's own; an off-request row is the
    # champion's, and it is replaced by whatever was crowned this time, so a
    # resumed or recomputed run never delivers two of them.
    requested = set(state["countries"])
    delivered = [row for row in results if row.get("country") in requested]
    if crowned.get("champion_row"):
        delivered.append(crowned["champion_row"])
    return {"champion": crowned, "mos_results": delivered}


async def evidence_node(state: dict) -> dict:
    """Assemble what the narrative is written from, and what the frontend renders.

    Deterministic and free: it reads stored results and collects nothing. No
    model is involved, which is what makes the narrative checkable afterwards --
    a claim that is not in this packet is not supported by anything.

    The packet comes back marked — requested, champion, display order, labels
    in the analysis language — and is delivered as it came.
    """
    entity_key = state.get("entity_key")
    if not entity_key:
        return {"breakdown": None}

    core = _client()
    await core.record_progress(state["analysis_id"], "assembling the evidence")
    packet = await core.build_evidence(
        entity_key,
        state["countries"],
        state["period_key"],
        state.get("demand_ruler"),
        state.get("brand_key"),
        language=state.get("language"),
    )
    return {"breakdown": packet}


async def deliver_node(state: dict) -> dict:
    """Record the answer, without waiting for the prose.

    The scores are the answer; the narrative is a reading of them. One model
    call once took longer than the entire rest of the analysis, so the request
    completes with its numbers and the text is attached a moment later.

    The rows go out as the scoring service computed them. Which ones were asked
    for, which is the champion and how each is shown are set where the record
    is written.
    """
    results = state.get("mos_results") or []
    requested = set(state["countries"])

    await _client().complete_analysis(
        state["analysis_id"], result=results, breakdown=state.get("breakdown")
    )

    # Measured here, where the buyer actually has the answer — not after the
    # narrative, which is attached a moment later and is not what was waited for.
    metrics.task_completion_rate(results, requested)
    metrics.response_latency_seconds(state.get("started_at"))
    # O consumo de tokens NÃO é medido aqui. O grafo é deliver -> narrate, e a
    # narrativa é a maior chamada de modelo da execução — mais o juiz e, desde
    # 23/09/2026, a revisão. Medir neste ponto reportava um total que não
    # incluía nada disso: numa execução real de 23/09 deu "0.0, no model call
    # in this run" com quatro chamadas feitas logo em seguida. Vai no fim de
    # tudo, em public._run, que é o único ponto por onde os dois caminhos
    # passam. A latência continua aqui de propósito: ela mede a espera do
    # comprador, que termina quando a resposta sai.
    metrics.tool_invocation_accuracy(state.get("coverage_tasks") or [], list(requested))
    observability.record("analysis.attempts", float(state.get("attempts", 1)))
    # How many markets were declared from the scoring service's record of no
    # recent demand instead of collected -- the measure that shows the skip
    # changing behaviour, not just existing.
    observability.record("absent_markets_skipped", float(len(state.get("absent_countries") or {})))
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
            packet,
            state.get("canonical_title") or "",
            state.get("user_prompt"),
            brand=state.get("brand_name"),
            language=state.get("language"),
        )
        await core.attach_narrative(state["analysis_id"], narrative)
        await core.record_progress(state["analysis_id"], None)
        await _check_narrative(narrative, packet, language=state.get("language"))
        return {"narrative": narrative}
    except (ScoringUnavailable, ScoringRejected) as exc:
        logger.exception("Analysis %s is delivered, but attaching its narrative failed", state["analysis_id"])
        observability.flag("WARNING", f"delivered without a narrative: {exc}")
        return {"errors": [f"narration failed: {exc}"]}


async def give_up_node(state: dict) -> dict:
    """End the run with a reason, rather than delivering part of a comparison.

    Which countries, and why each one. A failure nobody can read is the one the
    buyer asks about, and "it failed" does not say whether to try again in an
    hour or never.
    """
    missing = incomplete_countries(state)
    absent = state.get("absent_countries") or {}

    if _past_deadline(state):
        cause = "the sources stayed unavailable for the full retry window"
    elif _has_transient_failure(state, missing):
        cause = f"collection kept failing after {state.get('attempts', 0)} attempts"
    elif missing and all(country in absent for country in missing):
        # Not a fresh check: the buyer is owed the date the demand source last
        # answered, not "no data" repeated as if collection had just run.
        dates = sorted({absent[country][:10] for country in missing if absent[country]})
        cause = "no search demand when this was last checked" + (f", on {', '.join(dates)}" if dates else "")
    else:
        cause = _NO_DATA_CAUSE

    reason = f"Analysis incomplete for {', '.join(missing) or 'every requested market'}: {cause}."
    await _client().fail_analysis(state["analysis_id"], reason)
    # The run ends cleanly from here, so its trace would list as healthy.
    observability.flag("ERROR", reason)

    # Reported for a failed run too. A completion rate measured only on the runs
    # that completed is always 1.0, and a latency measured only on those hides
    # exactly the runs that took longest.
    requested = set(state["countries"])
    metrics.task_completion_rate(state.get("mos_results") or [], requested)
    metrics.response_latency_seconds(state.get("started_at"), what="given up")
    # Ver a nota em deliver_node: o consumo é reportado no fim, em public._run.
    metrics.tool_invocation_accuracy(state.get("coverage_tasks") or [], list(requested))
    observability.record("analysis.attempts", float(state.get("attempts", 1)))
    observability.record("absent_markets_skipped", float(len(absent)))
    return {"errors": [reason]}



async def _check_narrative(narrative: str | None, packet: dict[str, Any], *, language: str | None = None) -> None:
    """What the text claims, measured against what the evidence holds.

    Possible only because no number in the answer comes from a model: the
    narrative is written from a deterministic packet, so a figure that is not
    in the packet is not supported by anything. That check is exact and free,
    so it runs on every analysis.

    What it cannot reach — sentences that claim something without stating a
    figure — is the Langfuse evaluator's, which reads every `explanation`
    generation in the trace. The phrases the method forbids are the method's
    rules, in the narrative's language, so the scoring service checks those.

    Nothing here can fail the analysis. It is already delivered.
    """
    if not narrative:
        observability.record("narrative.written", False, comment="the model returned nothing")
        observability.flag("WARNING", "delivered without a narrative: the model returned nothing")
        return

    try:
        unsupported = quality.unsupported_numbers(narrative, packet)
        metrics.hallucination_rate(narrative, packet, unsupported)

        checked = await _client().language_check(narrative, language)
        violations = list(checked.get("violations") or [])
        observability.record(
            "narrative.language_violations",
            float(len(violations)),
            comment=("none" if not violations else "; ".join(violations)),
        )
    except Exception:  # noqa: BLE001 — a measurement must never break what it measures
        logger.exception("Could not check the narrative; the analysis is unaffected")
