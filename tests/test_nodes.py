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


def test_the_trace_says_which_sources_were_read_and_which_were_not(monkeypatch) -> None:
    """A diferença entre uma análise cara e uma barata não aparece no resultado:
    as duas têm a mesma forma. `skipped` é o dado que já estava no banco, dentro
    do prazo, sem nada pedido à fonte."""
    fatos = _fatos(monkeypatch)

    nodes._note_fontes(
        [
            {"source": "worldbank", "status": "skipped"},
            {"source": "worldbank", "status": "skipped"},
            {"source": "oecd", "status": "done"},
            {"source": "pytrends", "status": "failed"},
        ]
    )

    nome, resumo, detalhe = fatos[0]
    assert nome == "data_sources"
    assert "World Bank: 0 fetched, 2 reused from the database" in resumo
    assert "OECD: 1 fetched, 0 reused" in resumo
    assert "1 failed" in resumo
    assert detalhe["by_source"]["worldbank"] == {"fetched": 0, "reused": 2, "failed": 0}
    assert detalhe["fetched_total"] == 1


def test_an_analysis_that_downloaded_nothing_says_so(monkeypatch) -> None:
    """O caso que mais importa ver e que produz silêncio: tudo já estava aqui."""
    fatos = _fatos(monkeypatch)

    nodes._note_fontes([])

    assert "nothing was asked of any source" in fatos[0][1]


def test_the_trace_names_the_catalogue_entry_that_was_reused(monkeypatch) -> None:
    """Qual produto foi reaproveitado, e por qual dos passos — `trigram` decide
    por número, e o número vai junto."""
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
    assert "Picanha Swift" in resumo
    assert "reused an existing catalogue entry" in resumo
    assert "0.870" in resumo
    assert detalhe["reused"] is True


def test_a_brand_new_catalogue_entry_is_called_out_as_new(monkeypatch) -> None:
    """Só `created` significa produto inédito. Confundir isso com os outros três
    é confundir catálogo que funciona com catálogo que duplica."""
    fatos = _fatos(monkeypatch)

    nodes._note_catalogo({"canonical_title": "Picanha Bassi", "catalog_method": "created"})

    _, resumo, detalhe = fatos[0]
    assert "created a NEW catalogue entry" in resumo
    assert detalhe["reused"] is False


def test_an_analysis_from_before_the_field_existed_says_it_does_not_know(monkeypatch) -> None:
    """Dizer "não foi registrado" é melhor que inventar um método ou omitir o
    fato — as análises anteriores a 23/09/2026 não têm o campo."""
    fatos = _fatos(monkeypatch)

    nodes._note_catalogo({"canonical_title": "Air Fryer", "catalog_method": None})

    assert "was not recorded" in fatos[0][1]


def test_a_scoring_service_that_never_answered_is_not_called_a_cache_hit(monkeypatch) -> None:
    """`collect_node` fabrica uma tarefa sem fonte quando o scoring nem
    respondeu. Contar isso como "nada foi pedido porque tudo já estava no
    banco" é relatar sucesso onde houve queda — e a queda é justamente o que
    alguém abre o trace para encontrar."""
    fatos = _fatos(monkeypatch)

    nodes._note_fontes([{"country": "BRA", "status": "failed", "error_message": "connection refused"}])

    assert "no source was reached at all" in fatos[0][1]
    assert "already in the database" not in fatos[0][1]
