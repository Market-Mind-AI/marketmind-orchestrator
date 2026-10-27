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
            "language": "es",
        }
    )
    client.ensure_coverage = AsyncMock(return_value={"tasks": [], "search_terms": ["air fryer"]})
    client.demand_absence = AsyncMock(return_value={"markets": [], "window_days": 30})
    client.candidates_coverage = AsyncMock(return_value={"champion_cached": False})
    client.demand_ceiling = AsyncMock(return_value=(1_500_000.0, {"mode": "reference_candidate"}))
    client.compute_mos = AsyncMock(
        return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}]
    )
    client.crown_champion = AsyncMock(
        return_value={"champion_country": "ESP", "champion_mos": 70.0, "requested": True, "champion_row": None}
    )
    client.build_evidence = AsyncMock(return_value={"countries": [{"country": "ESP"}, {"country": "PRT"}]})
    client.narrate = AsyncMock(return_value="Spain leads.")
    client.language_check = AsyncMock(return_value={"violations": [], "language": "es"})
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


def _crowned_off_request(country: str = "BRA", mos: float = 88.0) -> dict:
    return {
        "champion_country": country,
        "champion_mos": mos,
        "requested": False,
        "champion_row": {"country": country, "mos": mos, "availability_status": "available"},
        "from_cache": True,
        "demand_ceiling": 1_500_000.0,
    }


async def test_the_champion_is_crowned_by_the_scoring_service(core: AsyncMock) -> None:
    """Deciding it here read the stored champion in one call and crowned it
    after the ruler had moved in the next. The scoring service holds the ruler,
    so it decides — with the requested rows and the run's period."""
    await build_graph().ainvoke({"analysis_id": "a1"})

    args = core.crown_champion.await_args.args
    assert args[0] == "e1"
    assert args[1] == nodes._period_key()
    assert args[2] == ["ESP", "PRT"]
    assert [r["country"] for r in args[3]] == ["ESP", "PRT"]


async def test_an_off_request_champion_is_delivered_as_its_row(core: AsyncMock) -> None:
    """The report draws the crown from the delivered rows. Naming the champion
    without its row crowned nothing (jamón ibérico, 23/10/2026)."""
    core.crown_champion = AsyncMock(return_value=_crowned_off_request())

    await build_graph().ainvoke({"analysis_id": "a1"})

    delivered = {row["country"]: row for row in core.complete_analysis.await_args.kwargs["result"]}
    assert delivered["BRA"]["mos"] == 88.0
    assert len(delivered) == 3, "the requested two plus the champion, once"


async def test_the_rows_and_the_packet_go_out_as_scoring_made_them(core: AsyncMock) -> None:
    """Which market was asked for and which is the champion are set where the
    record is written. Marking them here as well was a second answer to the
    same question, and the two disagreed on every rebuilt packet."""
    core.crown_champion = AsyncMock(return_value=_crowned_off_request())

    await build_graph().ainvoke({"analysis_id": "a1"})

    for row in core.complete_analysis.await_args.kwargs["result"]:
        assert "requested" not in row
        assert "is_champion" not in row
    packet = core.complete_analysis.await_args.kwargs["breakdown"]
    assert packet == {"countries": [{"country": "ESP"}, {"country": "PRT"}]}
    assert core.build_evidence.await_args.kwargs["language"] == "es"


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
    # In the language the record asked for — read back at the start, so a
    # resumed run writes in the same one.
    assert core.narrate.await_args.kwargs["language"] == "es"
    # A análise vai junto, para o texto aparecer na tela enquanto é escrito.
    assert core.narrate.await_args.kwargs["request_id"] == "a1"


async def test_any_failure_while_narrating_leaves_the_answer_delivered(core: AsyncMock) -> None:
    """Not only the scoring service's errors: anything else escaped to the
    run's own handler, which failed a record that was already done."""
    core.attach_narrative = AsyncMock(side_effect=KeyError("narrative"))

    final = await build_graph().ainvoke({"analysis_id": "a1"})

    core.fail_analysis.assert_not_awaited()
    assert final["delivered"] is True


async def test_a_failed_narration_leaves_no_stale_line_claiming_it_is_coming(core: AsyncMock) -> None:
    from app.scoring_client import ScoringUnavailable

    core.narrate = AsyncMock(side_effect=ScoringUnavailable("down"))

    final = await build_graph().ainvoke({"analysis_id": "a1"})

    assert any("narration failed" in e for e in final["errors"])
    core.complete_analysis.assert_awaited_once()


# --- A champion crowned again is replaced, not added ---


async def test_a_second_crowning_replaces_the_off_request_row(core: AsyncMock) -> None:
    """A resumed or recomputed run crowns again with the rows it already has;
    the old champion's row must not ride along next to the new one."""
    core.crown_champion = AsyncMock(return_value=_crowned_off_request("USA", 90.0))
    state = {
        "analysis_id": "a1",
        "entity_key": "e1",
        "countries": ["ESP", "PRT"],
        "period_key": "2026-10",
        "mos_results": [{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}, {"country": "BRA", "mos": 88.0}],
    }

    answer = await nodes.champion_node(state)

    assert [r["country"] for r in answer["mos_results"]] == ["ESP", "PRT", "USA"]


async def test_nothing_requested_scored_is_not_delivered_as_the_champion_alone(core: AsyncMock) -> None:
    core.crown_champion = AsyncMock(return_value=_crowned_off_request())
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



async def _executar(analysis_id: str = "a1") -> None:
    """O que `public._run` faz: o grafo dentro do bloco de contagem e do span.

    Os testes passaram a precisar disto em 23/09/2026, quando o consumo de
    tokens saiu do `deliver_node`. Ele era medido ali, antes da narrativa — que
    é a maior chamada de modelo da execução, mais o juiz e a revisão — e numa
    execução real reportou "0.0, no model call in this run" com quatro chamadas
    feitas logo depois. Passou a ser medido no fim, que é o único ponto por
    onde os dois caminhos do grafo passam.
    """
    from app import metrics, observability, spend

    with spend.tally(), observability.run_span(f"analysis {analysis_id}"):
        await build_graph().ainvoke({"analysis_id": analysis_id})
        metrics.token_consumption()


async def test_the_run_reports_the_five_measures(core: AsyncMock, caplog) -> None:
    """Every run says whether it did the job, how long the buyer waited, what it
    cost, how much of its work was worth doing, and how much of its text the
    evidence holds up. In the log, so one bad run can be read without a tracing
    backend, and on the trace, so many runs can be compared with one."""
    with caplog.at_level("INFO"):
        await _executar()

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


async def test_the_language_rules_are_checked_by_scoring_in_the_narratives_language(
    core: AsyncMock, caplog
) -> None:
    """A copy of the rules here was English-only: a Spanish narrative saying
    "demanda comprobada" passed unread."""
    core.narrate = AsyncMock(return_value="Hay demanda comprobada en España.")
    core.language_check = AsyncMock(
        return_value={"violations": ['"demanda comprobada": el interés de búsqueda no es demanda'], "language": "es"}
    )

    with caplog.at_level("INFO"):
        await build_graph().ainvoke({"analysis_id": "a1"})

    core.language_check.assert_awaited_once_with("Hay demanda comprobada en España.", "es")
    assert "narrative.language_violations=1.0" in caplog.text


async def test_a_language_check_that_fails_does_not_touch_the_delivered_answer(core: AsyncMock) -> None:
    from app.scoring_client import ScoringUnavailable

    core.language_check = AsyncMock(side_effect=ScoringUnavailable("down"))

    final = await build_graph().ainvoke({"analysis_id": "a1"})

    core.fail_analysis.assert_not_awaited()
    assert final["narrative"] == "Spain leads."


async def test_a_run_that_gave_up_reports_the_same_five(core: AsyncMock, caplog) -> None:
    """Measured on the runs that failed too. A completion rate read only off the
    runs that completed is always 1.0, and a latency read only off those hides
    exactly the runs that took longest — which are the ones worth seeing."""
    core.compute_mos = AsyncMock(
        return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}]
    )
    core.ensure_coverage = AsyncMock(return_value={"tasks": [{"country": "PRT", "status": "success"}]})

    with caplog.at_level("INFO"):
        await _executar()

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
    from app import observability

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

    # Exatamente o que public._run faz em produção, incluindo a contagem de
    # tokens no FIM — que é onde ela passou a ser feita.
    await _executar()

    assert registrados, "uma execução inteira não depositou nada no trace"
    fora = [nome for _, nome, dentro in registrados if not dentro]
    assert not fora, f"chegaram sem span aberto e seriam descartados ou viravam trace solto: {fora}"

    nomes = {nome for _, nome, _ in registrados}
    assert {"catalog_resolution", "data_sources"} <= nomes, "os fatos têm de estar no trace"
    assert {"task_completion_rate", "response_latency_seconds", "token_consumption"} <= nomes, (
        "as cinco medidas são o que o trace existe para comparar entre execuções"
    )


async def test_what_the_narrative_spends_is_counted(core: AsyncMock, caplog) -> None:
    """A regressão de 23/09/2026. O consumo era medido no `deliver_node`, e o
    grafo é deliver -> narrate: a narrativa é a MAIOR chamada de modelo da
    execução, mais o juiz e a revisão, e nada disso entrava na conta. Uma
    execução real reportou "token_consumption=0.0 (no model call in this run)"
    com quatro chamadas feitas logo em seguida.

    Aqui a narrativa gasta, como gasta em produção, e o total tem de vê-la.
    """
    import httpx

    from app import spend

    async def _narrar(*_args, **_kwargs) -> str:  # noqa: ANN002, ANN003
        spend.add_response(
            httpx.Headers({"X-LLM-Input-Tokens": "1200", "X-LLM-Output-Tokens": "800", "X-LLM-Calls": "1"})
        )
        return "Spain leads."

    core.narrate = _narrar

    with caplog.at_level("INFO"):
        await _executar()

    assert "QUALITY token_consumption=2000.0" in caplog.text, (
        "medido antes da narrativa, o total ignora a maior chamada da execução"
    )
    assert "no model call in this run" not in caplog.text
