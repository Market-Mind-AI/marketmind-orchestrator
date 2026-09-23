"""The whole flow, with the scoring service stubbed."""

from unittest.mock import AsyncMock

import pytest

from app import nodes
from app.graph import build_graph


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    client = AsyncMock()
    client.read_analysis = AsyncMock(
        return_value={
            "entity_key": "e1",
            "canonical_title": "air fryer",
            "countries": ["ESP", "PRT"],
            "user_prompt": "vale a pena?",
        }
    )
    client.ensure_coverage = AsyncMock(return_value={"tasks": [], "search_terms": ["air fryer"]})
    client.candidates_coverage = AsyncMock(return_value={"champion_cached": False})
    client.demand_ceiling = AsyncMock(return_value=(1_500_000.0, {"mode": "reference_candidate"}))
    client.compute_mos = AsyncMock(
        return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}]
    )
    client.score_candidates = AsyncMock(return_value=[])
    client.build_evidence = AsyncMock(return_value={"countries": [{"country": "ESP"}, {"country": "PRT"}]})
    client.narrate = AsyncMock(return_value="Spain leads.")
    monkeypatch.setattr(nodes, "_client", lambda: client)
    return client


async def test_a_complete_run_delivers_then_narrates(core: AsyncMock) -> None:
    """The scores are the answer and the prose is a reading of them, so the
    request completes first and the text is attached after."""
    final = await build_graph().ainvoke({"analysis_id": "a1"})

    core.complete_analysis.assert_awaited_once()
    core.attach_narrative.assert_awaited_once()
    core.fail_analysis.assert_not_awaited()
    assert final["narrative"] == "Spain leads."


async def test_the_delivered_rows_say_which_markets_were_asked_for(core: AsyncMock) -> None:
    core.compute_mos = AsyncMock(
        return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}]
    )
    core.score_candidates = AsyncMock(return_value=[{"country": "BRA", "mos": 88.0}])

    await build_graph().ainvoke({"analysis_id": "a1"})

    delivered = core.complete_analysis.await_args.kwargs["result"]
    by_country = {row["country"]: row for row in delivered}
    assert by_country["ESP"]["requested"] is True
    assert by_country["BRA"]["requested"] is False, "the champion is a benchmark, not a requested market"
    assert by_country["BRA"]["is_champion"] is True


async def test_a_cached_champion_skips_the_slowest_collection(core: AsyncMock) -> None:
    """Candidate demand re-run produces the same answer, and it is most of what
    a repeated analysis would otherwise pay for."""
    core.candidates_coverage = AsyncMock(
        return_value={"champion_cached": True, "champion_country": "BRA", "champion_mos": 88.0}
    )

    await build_graph().ainvoke({"analysis_id": "a1"})

    core.score_candidates.assert_not_awaited()
    core.set_champion.assert_not_awaited()


async def test_a_source_that_is_down_makes_the_run_go_back_and_collect(core: AsyncMock) -> None:
    core.compute_mos = AsyncMock(
        side_effect=[
            [{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}],
            [{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}],
        ]
    )
    core.ensure_coverage = AsyncMock(
        side_effect=[
            {"tasks": []},
            {"tasks": [{"country": "PRT", "status": "failed", "error_message": "502"}]},
            {"tasks": []},
            {"tasks": []},
        ]
    )

    await build_graph().ainvoke({"analysis_id": "a1"})

    assert core.compute_mos.await_count == 2, "it went back and scored again"
    core.complete_analysis.assert_awaited_once()
    core.fail_analysis.assert_not_awaited()


async def test_a_source_that_lacks_the_datum_ends_the_run_with_a_reason(core: AsyncMock) -> None:
    """Nothing partial goes out: a country missing from a ranking is not a
    smaller answer, it is a different one."""
    core.compute_mos = AsyncMock(
        return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}]
    )
    core.ensure_coverage = AsyncMock(return_value={"tasks": [{"country": "PRT", "status": "success"}]})

    await build_graph().ainvoke({"analysis_id": "a1"})

    core.complete_analysis.assert_not_awaited()
    core.fail_analysis.assert_awaited_once()
    reason = core.fail_analysis.await_args.args[1]
    assert "PRT" in reason
    assert "do not hold the data" in reason


async def test_a_failed_narration_does_not_undo_a_delivered_answer(core: AsyncMock) -> None:
    """An earlier version guarded only the model call, and a failure in the
    write marked an already-successful analysis as failed."""
    from app.scoring_client import ScoringUnavailable

    core.attach_narrative = AsyncMock(side_effect=ScoringUnavailable("down"))

    final = await build_graph().ainvoke({"analysis_id": "a1"})

    core.complete_analysis.assert_awaited_once()
    core.fail_analysis.assert_not_awaited()
    assert any("narration failed" in e for e in final["errors"])


async def test_the_evidence_packet_marks_the_champion_and_the_requested(core: AsyncMock) -> None:
    core.score_candidates = AsyncMock(return_value=[{"country": "BRA", "mos": 88.0}])

    await build_graph().ainvoke({"analysis_id": "a1"})

    packet = core.complete_analysis.await_args.kwargs["breakdown"]
    assert packet["champion_country"] == "BRA"
    assert packet["requested_countries"] == ["ESP", "PRT"]


async def test_the_run_keeps_saying_what_it_is_doing_to_the_very_end(core: AsyncMock) -> None:
    """It used to fall silent after scoring. Everything past that — crowning the
    champion, which can collect, assembling the evidence, writing the
    explanation — reported nothing, so the reader watched a stale line for the
    last stretch and then saw the answer appear with no warning."""
    await build_graph().ainvoke({"analysis_id": "a1"})

    said = [call.args[1] for call in core.record_progress.await_args_list]

    assert "scoring the markets" in said
    assert any("best market" in (s or "") for s in said), "crowning the champion can collect"
    assert any("evidence" in (s or "") for s in said)
    assert any("explanation" in (s or "") for s in said)


async def test_it_says_the_explanation_is_coming_before_writing_it(core: AsyncMock) -> None:
    """The request is already marked done by then, so without this the reader
    sees a finished analysis whose explanation is simply absent — with no way
    to tell "on its way" from "never arriving"."""
    await build_graph().ainvoke({"analysis_id": "a1"})

    said = [call.args[1] for call in core.record_progress.await_args_list]
    writing = said.index("writing the explanation")

    assert core.narrate.await_count == 1
    assert said[writing + 1] is None, "and cleared once it lands, so nothing stale is left behind"


async def test_a_failed_narration_leaves_no_stale_line_claiming_it_is_coming(core: AsyncMock) -> None:
    from app.scoring_client import ScoringUnavailable

    core.narrate = AsyncMock(side_effect=ScoringUnavailable("down"))

    final = await build_graph().ainvoke({"analysis_id": "a1"})

    assert any("narration failed" in e for e in final["errors"])
    core.complete_analysis.assert_awaited_once()


# --- The crown means "beats everything you asked about" ---


def _cached(country: str, mos: float) -> dict:
    return {"champion_cached": True, "champion_country": country, "champion_mos": mos}


async def test_a_cached_champion_that_still_wins_keeps_the_crown(core: AsyncMock) -> None:
    core.candidates_coverage = AsyncMock(return_value=_cached("BRA", 88.0))

    await build_graph().ainvoke({"analysis_id": "a1"})

    assert core.complete_analysis.await_args.kwargs["breakdown"]["champion_country"] == "BRA"
    core.set_champion.assert_not_awaited()


async def test_a_requested_market_that_beats_it_takes_the_crown(core: AsyncMock) -> None:
    """A champion stored at 66.37 stayed crowned in a later analysis where a
    requested market scored 69 — a visibly false claim, and the reason the
    comparison has to be re-run rather than the stored value trusted."""
    core.candidates_coverage = AsyncMock(return_value=_cached("BRA", 66.37))
    core.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 69.03}, {"country": "PRT", "mos": 40.0}])

    await build_graph().ainvoke({"analysis_id": "a1"})

    assert core.complete_analysis.await_args.kwargs["breakdown"]["champion_country"] == "ESP"


async def test_the_beaten_champion_is_replaced_not_merely_displaced(core: AsyncMock) -> None:
    """Fixing the current report is not enough. Left stored, the beaten
    champion is reloaded and crowned again on every later analysis — a request
    for two other markets would crown BRA at 66.37 while ESP at 69.03 was
    already known to be better."""
    core.candidates_coverage = AsyncMock(return_value=_cached("BRA", 66.37))
    core.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 69.03}, {"country": "PRT", "mos": 40.0}])

    await build_graph().ainvoke({"analysis_id": "a1"})

    core.set_champion.assert_awaited_once()
    stored = core.set_champion.await_args
    assert stored.kwargs["country"] == "ESP"
    assert stored.kwargs["mos"] == 69.03


async def test_a_tie_leaves_the_crown_with_the_requested_market(core: AsyncMock) -> None:
    """Strictly greater, so an equal stored champion adds nothing: the caller
    already has that market in their own list."""
    core.candidates_coverage = AsyncMock(return_value=_cached("BRA", 70.0))
    core.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 40.0}])

    await build_graph().ainvoke({"analysis_id": "a1"})

    assert core.complete_analysis.await_args.kwargs["breakdown"]["champion_country"] == "ESP"
    core.set_champion.assert_awaited_once()


async def test_a_cached_champion_wins_when_nothing_requested_scored(core: AsyncMock) -> None:
    core.candidates_coverage = AsyncMock(return_value=_cached("BRA", 66.37))
    core.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": None}, {"country": "PRT", "mos": None}])
    core.ensure_coverage = AsyncMock(return_value={"tasks": [{"country": "ESP", "status": "success"}]})

    await build_graph().ainvoke({"analysis_id": "a1"})

    # Nothing requested could be scored, so the run gives up rather than
    # delivering a ranking made only of a market nobody asked about.
    core.fail_analysis.assert_awaited_once()


# --- what the run reports about itself ---


async def test_the_search_terms_are_logged_per_country(core: AsyncMock, caplog) -> None:
    """The one thing a model chose that nothing downstream exposes. Get the
    native term wrong and the analysis is immaculate and about the wrong
    words — with no symptom anywhere else."""
    core.ensure_coverage = AsyncMock(
        return_value={
            "tasks": [],
            "location": "ES",
            "language": "es",
            "search_terms": ["freidora de aire", "air fryer"],
            "category_terms": ["freidora"],
        }
    )

    with caplog.at_level("INFO"):
        await build_graph().ainvoke({"analysis_id": "a1"})

    assert "SEARCH TERMS ESP (ES, es)" in caplog.text
    assert "freidora de aire" in caplog.text


async def test_the_run_reports_the_five_measures(core: AsyncMock, caplog) -> None:
    """Every run says whether it did the job, how long the buyer waited, what it
    cost, how much of its work was worth doing, and how much of its text the
    evidence holds up. In the log, so one bad run can be read without a tracing
    backend, and on the trace, so many runs can be compared with one."""
    with caplog.at_level("INFO"):
        await build_graph().ainvoke({"analysis_id": "a1"})

    assert "QUALITY task_completion_rate=1.0 (2 of 2 requested markets scored)" in caplog.text
    assert "QUALITY response_latency_seconds=" in caplog.text
    assert "QUALITY token_consumption=" in caplog.text
    assert "QUALITY tool_invocation_accuracy=1.0" in caplog.text
    assert "QUALITY hallucination_rate=" in caplog.text


async def test_a_redundant_collection_is_counted(core: AsyncMock, caplog) -> None:
    """Where the cost leaks: the same country collected twice in one run."""
    core.ensure_coverage = AsyncMock(
        side_effect=[
            {"tasks": [{"country": "ESP"}, {"country": "ESP"}]},  # collected twice
            {"tasks": [{"country": "PRT"}]},
        ]
    )

    with caplog.at_level("INFO"):
        await build_graph().ainvoke({"analysis_id": "a1"})

    # 3 collection calls, 1 of them the same country twice.
    assert "QUALITY tool_invocation_accuracy=0.6667" in caplog.text
    assert "1 redundant, 0 off-request" in caplog.text


async def test_a_figure_the_evidence_does_not_hold_is_reported(core: AsyncMock, caplog) -> None:
    """Checkable only because no number in the answer comes from a model."""
    core.build_evidence = AsyncMock(return_value={"countries": [{"country": "ESP", "mos": 40.02}]})
    core.narrate = AsyncMock(return_value="Spain scores 40 with 9,400,000 searches a month.")

    with caplog.at_level("INFO"):
        await build_graph().ainvoke({"analysis_id": "a1"})

    # One figure stated that the evidence does not hold, out of two stated.
    assert "QUALITY hallucination_rate=0.5" in caplog.text
    assert "9400000" in caplog.text


async def test_a_narrative_that_obeys_the_rules_reports_nothing(core: AsyncMock, caplog) -> None:
    core.build_evidence = AsyncMock(return_value={"countries": [{"country": "ESP", "mos": 40.02}]})
    core.narrate = AsyncMock(return_value="Search interest in Spain is moderate, scoring 40.")

    with caplog.at_level("INFO"):
        await build_graph().ainvoke({"analysis_id": "a1"})

    assert "QUALITY hallucination_rate=0.0" in caplog.text
    assert "narrative.language_violations=0.0" in caplog.text


async def test_with_no_judge_configured_the_run_says_so_and_carries_on(
    core: AsyncMock, caplog, monkeypatch
) -> None:
    """Asked for explicitly: do not run that part, and say it is not
    configured — rather than failing, or leaving a silent gap."""
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("JUDGE_SAMPLE_RATE", "1.0")

    with caplog.at_level("INFO"):
        await build_graph().ainvoke({"analysis_id": "a1"})

    assert "judge is not configured" in caplog.text
    assert "narrative.unsupported_claims" not in caplog.text, (
        "no score at all — an absent judgement must not read as a clean one"
    )
    core.complete_analysis.assert_awaited_once()


async def test_a_run_that_gave_up_reports_the_same_five(core: AsyncMock, caplog) -> None:
    """Measured on the runs that failed too. A completion rate read only off the
    runs that completed is always 1.0, and a latency read only off those hides
    exactly the runs that took longest — which are the ones worth seeing."""
    core.compute_mos = AsyncMock(
        return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}]
    )
    core.ensure_coverage = AsyncMock(return_value={"tasks": [{"country": "PRT", "status": "success"}]})

    with caplog.at_level("INFO"):
        await build_graph().ainvoke({"analysis_id": "a1"})

    core.fail_analysis.assert_awaited_once()
    assert "QUALITY task_completion_rate=0.5 (1 of 2 requested markets scored)" in caplog.text
    assert "QUALITY response_latency_seconds=" in caplog.text
    assert "run start -> given up" in caplog.text
    assert "QUALITY token_consumption=" in caplog.text
    assert "QUALITY tool_invocation_accuracy=" in caplog.text


# --- o que uma execução inteira deposita no trace (23/09/2026) --------------


async def test_every_measure_of_a_full_run_lands_inside_the_trace(core, monkeypatch) -> None:  # noqa: ANN001
    """O teste que faltava, e a razão de faltar é o próprio defeito.

    Em 23/09/2026 descobriu-se que as cinco métricas nunca tinham chegado ao
    Langfuse: cada linha QUALITY era seguida de "No active span in current
    context... will be skipped". Passavam em todo teste porque a linha do log
    sai primeiro e incondicionalmente, e era nela que os testes batiam.

    Os fatos falhavam de um jeito pior: `create_event` não descarta, cria um
    TRACE RAIZ próprio — estariam no Langfuse, soltos da análise que os
    produziu, presentes e inacháveis.

    Por isso este teste não olha texto de log. Ele afirma que cada medição
    aconteceu com um span ABERTO, que é a condição de chegar ao lugar certo.
    """
    from app import observability, quality

    aberto = {"span": False}
    registrados: list[tuple[str, str, bool]] = []

    class _SpanFalso:
        def __enter__(self):  # noqa: ANN204
            aberto["span"] = True
            return self

        def __exit__(self, *_):  # noqa: ANN002, ANN204
            aberto["span"] = False
            return False

    class _ClienteFalso:
        def start_as_current_observation(self, **_):  # noqa: ANN003, ANN201
            return _SpanFalso()

        def create_event(self, *, name, output=None, metadata=None):  # noqa: ANN001, ANN201
            registrados.append(("event", name, aberto["span"]))

        def score_current_trace(self, **kwargs):  # noqa: ANN003, ANN201
            registrados.append(("score", kwargs.get("name"), aberto["span"]))

    monkeypatch.setattr(observability, "_handler", lambda: object())
    monkeypatch.setattr("langfuse.get_client", lambda: _ClienteFalso())
    monkeypatch.setattr(quality, "judge_support", AsyncMock(return_value=None))

    # Exatamente o que public._run faz em produção.
    with observability.run_span("analysis a1"):
        await build_graph().ainvoke({"analysis_id": "a1"})

    assert registrados, "uma execução inteira não depositou nada no trace"
    fora = [nome for _, nome, dentro in registrados if not dentro]
    assert not fora, f"chegaram sem span aberto e seriam descartados ou viravam trace solto: {fora}"

    nomes = {nome for _, nome, _ in registrados}
    assert {"catalog_resolution", "data_sources"} <= nomes, "os fatos têm de estar no trace"
    assert {"task_completion_rate", "response_latency_seconds", "token_consumption"} <= nomes, (
        "as cinco medidas são o que o trace existe para comparar entre execuções"
    )
