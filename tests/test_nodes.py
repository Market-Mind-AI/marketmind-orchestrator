"""The decision the analysis turns on: insist, give up, or deliver."""

from datetime import UTC, datetime, timedelta

import pytest

from app import nodes

_REQUESTED = ["ESP", "PRT", "BRA"]


def _state(**overrides) -> dict:
    base = {
        "analysis_id": "a1",
        "entity_key": "e1",
        "countries": list(_REQUESTED),
        "period_key": "2026-09",
        "attempts": 1,
        "coverage_tasks": [],
        "mos_results": [],
        "deadline_at": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
    }
    base.update(overrides)
    return base


def _scored(*countries: str) -> list[dict]:
    return [{"country": c, "mos": 70.0} for c in countries]


# --- what counts as done ---


def test_every_requested_market_scored_is_complete() -> None:
    assert nodes.after_scoring(_state(mos_results=_scored(*_REQUESTED))) == "complete"


def test_a_row_without_a_score_is_the_same_gap_as_no_row() -> None:
    """A country measured but unrankable is not a country delivered."""
    results = [*_scored("ESP", "PRT"), {"country": "BRA", "mos": None}]

    assert nodes.incomplete_countries(_state(mos_results=results)) == ["BRA"]


def test_an_off_request_market_does_not_make_the_request_complete() -> None:
    """The champion is a benchmark, not a substitute for what was asked."""
    results = [*_scored("ESP", "PRT"), {"country": "JPN", "mos": 80.0}]

    assert nodes.incomplete_countries(_state(mos_results=results)) == ["BRA"]


# --- insisting, and when to stop ---


def test_a_source_that_is_down_is_worth_insisting_on() -> None:
    """The August outage: nine of twenty-four countries dropped out of a ranking
    because a source was unreachable for twenty-nine minutes."""
    state = _state(
        mos_results=_scored("ESP", "PRT"),
        coverage_tasks=[{"country": "BRA", "status": "failed", "error_message": "502"}],
    )

    assert nodes.after_scoring(state) == "insist"


def test_a_source_that_simply_lacks_the_datum_is_not() -> None:
    """A collection that finished without failing and still left the indicator
    missing will produce the same nothing however many times it is asked."""
    state = _state(
        mos_results=_scored("ESP", "PRT"),
        coverage_tasks=[{"country": "BRA", "status": "success"}],
    )

    assert nodes.after_scoring(state) == "give_up"


def test_insisting_stops_at_the_attempt_limit() -> None:
    state = _state(
        attempts=nodes.MAX_ATTEMPTS,
        mos_results=_scored("ESP", "PRT"),
        coverage_tasks=[{"country": "BRA", "status": "failed"}],
    )

    assert nodes.after_scoring(state) == "give_up"


def test_insisting_stops_at_the_deadline() -> None:
    """Past the window the run fails with a reason rather than delivering part
    of a comparison."""
    state = _state(
        deadline_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        mos_results=_scored("ESP", "PRT"),
        coverage_tasks=[{"country": "BRA", "status": "failed"}],
    )

    assert nodes.after_scoring(state) == "give_up"


def test_a_complete_run_never_gives_up_even_past_the_deadline() -> None:
    """The deadline bounds insisting, not delivering. Work already finished is
    not thrown away because the clock ran out while it finished."""
    state = _state(
        deadline_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        mos_results=_scored(*_REQUESTED),
    )

    assert nodes.after_scoring(state) == "complete"


# --- the period is decided once ---


def test_the_period_is_the_calendar_month_in_utc() -> None:
    assert nodes._period_key(datetime(2026, 9, 21, 23, 59, tzinfo=UTC)) == "2026-09"
    assert nodes._period_key(datetime(2026, 1, 1, 0, 0, tzinfo=UTC)) == "2026-01"


async def test_a_resumed_run_keeps_the_period_it_started_with(monkeypatch: pytest.MonkeyPatch) -> None:
    """An analysis that begins in September and resumes in October must not
    silently change which month it is measuring."""
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.read_analysis = AsyncMock(return_value={"entity_key": "e1", "countries": ["ESP"]})
    monkeypatch.setattr(nodes, "_client", lambda: client)

    resumed = await nodes.start_node({"analysis_id": "a1", "period_key": "2026-09", "attempts": 2})

    assert resumed["period_key"] == "2026-09"
    assert resumed["attempts"] == 2


async def test_a_fresh_run_decides_the_period_and_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.read_analysis = AsyncMock(return_value={"entity_key": "e1", "countries": ["ESP"]})
    monkeypatch.setattr(nodes, "_client", lambda: client)

    started = await nodes.start_node({"analysis_id": "a1"})

    assert started["period_key"] == nodes._period_key()
    assert started["deadline_at"] is not None
    assert started["attempts"] == 0


# --- o que o trace passa a dizer (23/09/2026) -------------------------------


def _fatos(monkeypatch) -> list[tuple[str, str, dict]]:
    """Captura o que seria anexado ao trace, sem precisar de um."""
    capturados: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        nodes.observability,
        "note",
        lambda nome, *, summary, detail=None: capturados.append((nome, summary, detail or {})),
    )
    monkeypatch.setattr(nodes.observability, "record", lambda *a, **k: None)
    return capturados


def test_every_source_is_named_even_when_it_did_nothing(monkeypatch) -> None:
    """A regressão que motivou o formato, medida em 23/09/2026: a primeira
    execução real relatou só "Google Ads: 2 fetched" e não disse uma palavra
    sobre World Bank, OECD e UN Data — que tinham servido TODOS os indicadores
    macro a partir do banco. Dado fresco não vira tarefa, e a versão anterior
    só contava tarefas. Sumir da linha é indistinguível de não ter sido
    consultado, que é justamente o que a linha existe para responder."""
    fatos = _fatos(monkeypatch)

    nodes._note_fontes(
        [{"source": "pytrends", "status": "done"}, {"source": "pytrends", "status": "done"}],
        reused={"worldbank": 12, "oecd": 6, "undata": 4},
    )

    _, resumo, detalhe = fatos[0]
    assert "World Bank: 0 fetched, 12 reused from the database" in resumo
    assert "OECD: 0 fetched, 6 reused" in resumo
    assert "UN Data: 0 fetched, 4 reused" in resumo
    assert "Google Ads (via DataForSEO): 2 fetched, 0 reused" in resumo
    assert detalhe["worldbank"] == {"fetched": 0, "reused": 12, "failed": 0}


def test_a_source_that_does_not_cover_the_country_says_so(monkeypatch) -> None:
    """"Não cobre este país" é uma resposta, e é diferente de "não precisei
    dela". A OCDE não publica preço imobiliário para Argentina e Peru, e isso
    se lê do mapeamento do país, não de uma lista mantida à mão."""
    fatos = _fatos(monkeypatch)

    nodes._note_fontes([], reused={"worldbank": 12}, not_covering={"oecd": ["ARG", "PER"]})

    _, resumo, detalhe = fatos[0]
    assert "OECD: does not cover ARG, PER" in resumo
    assert detalhe["oecd"]["not_covering"] == ["ARG", "PER"]


def test_a_source_nothing_was_needed_from_is_distinguished_from_the_others(monkeypatch) -> None:
    """Três silêncios diferentes, e confundi-los custa caro: aproveitado do
    banco, não cobre o país, e nada desta análise vem dessa fonte."""
    fatos = _fatos(monkeypatch)

    nodes._note_fontes([], reused={"worldbank": 12})

    resumo = fatos[0][1]
    assert "World Bank: 0 fetched, 12 reused" in resumo
    assert "OECD: nothing needed from it" in resumo


def test_a_failed_collection_is_counted_against_its_own_source(monkeypatch) -> None:
    fatos = _fatos(monkeypatch)

    nodes._note_fontes([{"source": "undata", "status": "failed"}], reused={})

    resumo = fatos[0][1]
    assert "UN Data: 0 fetched, 0 reused from the database, 1 failed" in resumo


def test_a_scoring_service_that_never_answered_is_not_called_a_cache_hit(monkeypatch) -> None:
    """`collect_node` fabrica uma tarefa sem fonte quando o scoring nem
    respondeu. Contar isso como reaproveitamento é relatar sucesso onde houve
    queda — e a queda é justamente o que alguém abre o trace para encontrar."""
    fatos = _fatos(monkeypatch)

    nodes._note_fontes([{"country": "BRA", "status": "failed", "error_message": "connection refused"}])

    resumo = fatos[0][1]
    assert "could not even run" in resumo
    assert "the scoring service did not answer" in resumo


def test_a_spelling_match_reports_the_percentage_and_the_reuse_flag(monkeypatch) -> None:
    """Três coisas numa linha: coincidiu, de que tipo, e quanto. O trigram é o
    único passo que decide por número, e o número vira porcentagem porque é
    assim que se lê "quanto coincidiu"."""
    fatos = _fatos(monkeypatch)

    nodes._note_catalogo(
        {
            "canonical_title": "Picanha Swift",
            "entity_key": "e1",
            "catalog_method": "trigram",
            "catalog_match_score": 0.87,
        }
    )

    nome, resumo, detalhe = fatos[0]
    assert nome == "catalog_resolution"
    assert "REUSED the catalogue entry" in resumo
    assert "match by spelling" in resumo
    assert "87.0% similar" in resumo
    assert detalhe["reused_existing_entry"] is True
    assert detalhe["match_percent"] == 87.0
    assert detalhe["match_type"] == "spelling"


def test_a_meaning_match_has_no_percentage_and_says_why(monkeypatch) -> None:
    """A coincidência SEMÂNTICA é a que não tem número: `quadcopter` e `drone`
    não têm semelhança de escrita nenhuma. Foi essa a razão de o passo de vetor
    sair em 14/08/2026 — nenhuma distância de cosseno separa esses casos,
    porque a relação é conhecimento. O que explica a decisão é a frase."""
    fatos = _fatos(monkeypatch)

    nodes._note_catalogo(
        {
            "canonical_title": "drone",
            "catalog_method": "llm",
            "catalog_match_score": None,
            "catalog_match_reasoning": "quadcopter is a synonym for drone",
        }
    )

    _, resumo, detalhe = fatos[0]
    assert "REUSED the catalogue entry" in resumo
    assert "match by meaning" in resumo
    assert "quadcopter is a synonym for drone" in resumo
    assert "% similar" not in resumo, "não há porcentagem quando a coincidência é de significado"
    assert detalhe["match_percent"] is None
    assert detalhe["reused_existing_entry"] is True


def test_an_exact_alias_is_not_confused_with_a_similarity(monkeypatch) -> None:
    fatos = _fatos(monkeypatch)

    nodes._note_catalogo({"canonical_title": "picanha", "catalog_method": "alias"})

    _, resumo, detalhe = fatos[0]
    assert "match by exact" in resumo
    assert detalhe["match_type"] == "exact"
    assert detalhe["match_percent"] is None


def test_a_brand_new_catalogue_entry_is_called_out_as_new(monkeypatch) -> None:
    """Só `created` significa produto inédito. Confundir isso com os outros três
    é confundir catálogo que funciona com catálogo que duplica."""
    fatos = _fatos(monkeypatch)

    nodes._note_catalogo({"canonical_title": "Picanha Bassi", "catalog_method": "created"})

    _, resumo, detalhe = fatos[0]
    assert "CREATED a new catalogue entry" in resumo
    assert "match by none" in resumo
    assert detalhe["reused_existing_entry"] is False


def test_an_analysis_from_before_the_field_existed_says_it_does_not_know(monkeypatch) -> None:
    """Dizer "não foi registrado" é melhor que inventar um método ou omitir o
    fato — as análises anteriores a 23/09/2026 não têm o campo."""
    fatos = _fatos(monkeypatch)

    nodes._note_catalogo({"canonical_title": "Air Fryer", "catalog_method": None})

    assert "was not recorded" in fatos[0][1]


def test_demand_that_was_entirely_in_the_database_is_not_called_unneeded(monkeypatch) -> None:
    """Medido em 23/09/2026, na segunda execução real: a linha dizia "Google
    Ads: nothing needed from it" quando a verdade era que TODOS os termos e as
    palavras relacionadas já estavam guardados e dentro dos 30 dias. Isso é
    reaproveitamento — o caso que custa dinheiro confundir — e não ausência."""
    fatos = _fatos(monkeypatch)

    nodes._note_fontes([], reused={"worldbank": 24}, related_fresh=2)

    resumo = fatos[0][1]
    assert "nothing fetched — every search term and related keyword was already stored" in resumo
    assert "nothing needed from it" not in resumo.split("Google Ads")[1]
