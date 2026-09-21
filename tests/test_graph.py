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
    client.ensure_coverage = AsyncMock(return_value=[])
    client.candidates_coverage = AsyncMock(return_value=False)
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
    core.candidates_coverage = AsyncMock(return_value=True)

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
            [],
            [{"country": "PRT", "status": "failed", "error_message": "502"}],
            [],
            [],
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
    core.ensure_coverage = AsyncMock(return_value=[{"country": "PRT", "status": "success"}])

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
