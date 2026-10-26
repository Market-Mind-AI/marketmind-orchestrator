"""The agent's decision point: what it may choose, what it chose, and that the
choice is carried out by the graph's own nodes — with the model mocked, and
without it.
"""

from unittest.mock import AsyncMock

import pytest

from app import agent, nodes
from app.agent import Action, Situation
from app.graph import build_graph
from app.scoring_client import ScoringUnavailable


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    client = AsyncMock()
    client.read_analysis = AsyncMock(
        return_value={"entity_key": "e1", "canonical_title": "air fryer", "countries": ["ESP", "PRT"], "language": "en"}
    )
    client.demand_absence = AsyncMock(return_value={"markets": [], "window_days": 30})
    client.ensure_coverage = AsyncMock(return_value={"tasks": []})
    client.candidates_coverage = AsyncMock(return_value={"champion_cached": True})
    client.demand_ceiling = AsyncMock(return_value=(15_800.0, {"mode": "reference"}))
    client.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}])
    client.crown_champion = AsyncMock(
        return_value={"champion_country": "ESP", "requested": True, "champion_row": None, "demand_ceiling": 15_800.0}
    )
    client.build_evidence = AsyncMock(return_value={"countries": [{"country": "ESP"}, {"country": "PRT"}]})
    client.narrate = AsyncMock(return_value="Spain leads.")
    client.language_check = AsyncMock(return_value={"violations": [], "language": "en"})
    monkeypatch.setattr(nodes, "_client", lambda: client)
    return client


class _Model:
    """A stand-in for the model: answers in order, and remembers what it was
    offered."""

    def __init__(self, *answers: tuple[str, str]) -> None:
        self.answers = list(answers)
        self.asked: list[tuple[Situation, dict, list[Action]]] = []

    async def __call__(self, situation: Situation, facts: dict, allowed: list[Action]) -> tuple[str, str]:
        self.asked.append((situation, facts, allowed))
        return self.answers.pop(0)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    def install(*answers: tuple[str, str]) -> _Model:
        fake = _Model(*answers)
        monkeypatch.setattr(agent, "_chooser", lambda: fake)
        return fake

    return install


def _progress(core: AsyncMock) -> list[str]:
    return [call.args[1] for call in core.record_progress.await_args_list if call.args[1]]


# --- (a) a country's collection failed after the scoring service's retries ---


def _prt_collection_failed(core: AsyncMock) -> None:
    core.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}])
    core.ensure_coverage = AsyncMock(
        return_value={
            "tasks": [{"country": "PRT", "source": "oecd", "status": "failed", "error_message": "oecd ingestion run ended with status=failed"}]
        }
    )


async def test_a_failed_collection_the_agent_continues_without(core: AsyncMock, model) -> None:
    _prt_collection_failed(core)
    fake = model(("continue_without", "the OECD run failed for Portugal and Spain answers the question"))

    final = await build_graph().ainvoke({"analysis_id": "a1"})

    situation, facts, allowed = fake.asked[0]
    assert situation is Situation.COLLECTION_FAILED
    assert facts["missing_countries"] == ["PRT"]
    assert Action.CONTINUE_WITHOUT in allowed and Action.WAIT_AND_RESUME not in allowed
    core.complete_analysis.assert_awaited_once()
    core.fail_analysis.assert_not_awaited()
    # The reason is on the record the buyer watches, and in the run's state.
    assert any(line.startswith("agent decision: continue without — the OECD run failed") for line in _progress(core))
    assert final["decisions"][0]["decided_by"] == "model"


async def test_a_failed_collection_the_agent_retries_goes_back_to_collect(core: AsyncMock, model) -> None:
    _prt_collection_failed(core)
    core.compute_mos = AsyncMock(
        side_effect=[
            [{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}],
            [{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}],
        ]
    )
    model(("retry_later", "a single failed run is worth one more try"))

    await build_graph().ainvoke({"analysis_id": "a1"})

    assert core.compute_mos.await_count == 2
    core.complete_analysis.assert_awaited_once()


# --- (b) a source is down --------------------------------------------------


async def test_a_scoring_timeout_while_scoring_is_recoverable(core: AsyncMock, model, monkeypatch) -> None:
    """A timeout here used to fail the whole run on the spot."""
    core.compute_mos = AsyncMock(
        side_effect=[
            ScoringUnavailable("POST /mos/compute failed: ReadTimeout"),
            [{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}],
        ]
    )
    slept: list[float] = []
    real_sleep = nodes.asyncio.sleep

    async def short_sleep(seconds: float) -> None:
        slept.append(seconds)
        await real_sleep(min(seconds, 0.01))

    # One second of waiting, slept in hundredths: the loop is real, the clock
    # only short.
    monkeypatch.setattr(nodes.asyncio, "sleep", short_sleep)
    monkeypatch.setattr(nodes, "get_settings", lambda: _settings(source_wait_seconds=1))
    fake = model(("wait_and_resume", "the scoring service timed out; it usually recovers within minutes"))

    await build_graph().ainvoke({"analysis_id": "a1"})

    assert fake.asked[0][0] is Situation.SOURCE_DOWN
    assert Action.WAIT_AND_RESUME in fake.asked[0][2]
    assert slept, "it waited before going back"
    core.complete_analysis.assert_awaited_once()
    core.fail_analysis.assert_not_awaited()


async def test_a_source_down_the_agent_delivers_with_a_note(core: AsyncMock, model) -> None:
    core.compute_mos = AsyncMock(return_value=[{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": None}])
    core.ensure_coverage = AsyncMock(
        return_value={"tasks": [{"country": "PRT", "source": "pytrends", "status": "failed", "error_message": "503 Service Unavailable"}]}
    )
    model(("deliver_with_note", "Google Ads is down and the buyer is better served by Spain now"))

    await build_graph().ainvoke({"analysis_id": "a1"})

    breakdown = core.complete_analysis.await_args.kwargs["breakdown"]
    assert breakdown["delivery_notes"] == [{"kind": "source_down", "countries_without_score": ["PRT"]}]
    # The packet itself is the scoring service's, untouched.
    assert breakdown["countries"] == [{"country": "ESP"}, {"country": "PRT"}]


# --- (c) an incoherent result before delivery ------------------------------


async def test_a_ruler_that_moved_gets_one_recompute_then_delivers(core: AsyncMock, model) -> None:
    """A requested market above every reference market raises the ruler after
    the markets were measured against the old one (Brazil over the USA,
    26/10/2026)."""
    core.crown_champion = AsyncMock(
        side_effect=[
            {"champion_country": "ESP", "requested": True, "champion_row": None, "demand_ceiling": 17_020.0},
            {"champion_country": "ESP", "requested": True, "champion_row": None, "demand_ceiling": 17_020.0},
        ]
    )
    core.demand_ceiling = AsyncMock(
        side_effect=[(15_800.0, {"mode": "reference"}), (17_020.0, {"mode": "reference"})]
    )
    fake = model(("recompute_once", "the ruler moved after measuring; measure again against the new one"))

    await build_graph().ainvoke({"analysis_id": "a1"})

    assert fake.asked[0][0] is Situation.INCOHERENT
    assert fake.asked[0][1]["incoherences"] == ["ruler_moved_after_measuring"]
    assert core.demand_ceiling.await_count == 2, "it went back to the ruler"
    assert core.compute_mos.await_count == 2
    core.complete_analysis.assert_awaited_once()


async def test_a_second_incoherence_cannot_recompute_again(core: AsyncMock, model) -> None:
    core.crown_champion = AsyncMock(
        return_value={"champion_country": "BRA", "requested": False, "champion_row": None, "demand_ceiling": 15_800.0}
    )
    fake = model(
        ("recompute_once", "the champion row did not come back"),
        ("deliver_with_note", "still missing after one rebuild; deliver what holds"),
    )

    await build_graph().ainvoke({"analysis_id": "a1"})

    assert Action.RECOMPUTE_ONCE in fake.asked[0][2]
    assert Action.RECOMPUTE_ONCE not in fake.asked[1][2], "one recompute, not a loop"
    notes = core.complete_analysis.await_args.kwargs["breakdown"]["delivery_notes"]
    assert notes[0]["kind"] == "incoherent"
    assert "champion_row_missing:BRA" in notes[0]["incoherences"]


def test_a_requested_market_missing_from_the_evidence_is_incoherent() -> None:
    state = {
        "countries": ["ESP", "PRT"],
        "mos_results": [{"country": "ESP", "mos": 70.0}, {"country": "PRT", "mos": 61.0}],
        "breakdown": {"countries": [{"country": "ESP"}]},
        "champion": {"champion_country": "ESP", "requested": True, "demand_ceiling": 1.0},
        "demand_ceiling": 1.0,
    }

    assert nodes.check_result(state)["incoherence"] == ["requested_country_missing_from_evidence:PRT"]


def test_a_coherent_result_goes_straight_to_delivery() -> None:
    state = {
        "countries": ["ESP"],
        "mos_results": [{"country": "ESP", "mos": 70.0}, {"country": "BRA", "mos": 80.0}],
        "breakdown": {"countries": [{"country": "BRA"}, {"country": "ESP"}]},
        "champion": {"champion_country": "BRA", "requested": False, "champion_row": {"country": "BRA"}, "demand_ceiling": None},
        "demand_ceiling": None,
    }

    assert nodes.check_result(state)["incoherence"] is None
    assert nodes.after_check({"incoherence": None}) == "deliver"


# --- the fence and the fallback --------------------------------------------


async def test_without_a_model_the_fixed_rule_decides(core: AsyncMock) -> None:
    """No key: today's behaviour, insist then give up — and it says so."""
    _prt_collection_failed(core)

    final = await build_graph().ainvoke({"analysis_id": "a1"})

    assert {d["decided_by"] for d in final["decisions"]} <= {"fallback", "only_option"}
    assert final["decisions"][0]["action"] == "retry_later"
    core.fail_analysis.assert_awaited_once()
    core.complete_analysis.assert_not_awaited()


async def test_an_action_outside_the_fence_is_not_taken(model) -> None:
    model(("deliver", "deliver anyway"))

    decision = await agent.decide(
        Situation.COLLECTION_FAILED, {}, [Action.RETRY_LATER, Action.GIVE_UP]
    )

    assert decision.action is Action.RETRY_LATER
    assert decision.decided_by == "fallback"


async def test_a_model_that_fails_falls_back_to_the_rule(monkeypatch) -> None:
    async def broken(*_args):
        raise RuntimeError("overloaded")

    monkeypatch.setattr(agent, "_chooser", lambda: broken)

    decision = await agent.decide(Situation.SOURCE_DOWN, {}, [Action.WAIT_AND_RESUME, Action.GIVE_UP])

    assert decision.action is Action.GIVE_UP
    assert decision.decided_by == "fallback"


async def test_a_single_option_does_not_ask_the_model(model) -> None:
    fake = model(("retry_later", "x"))

    decision = await agent.decide(Situation.SOURCE_DOWN, {}, [Action.GIVE_UP])

    assert decision.action is Action.GIVE_UP and decision.decided_by == "only_option"
    assert fake.asked == []


def test_a_reason_with_a_figure_stays_off_the_buyers_screen() -> None:
    """Every number a buyer reads comes from the scoring service."""
    decision = agent.Decision(Action.DELIVER_WITH_NOTE, "Spain scored 70, good enough", "model")

    assert "70" not in agent.for_progress(decision)


def test_nothing_is_delivered_when_nothing_requested_scored() -> None:
    state = {
        "countries": ["ESP", "PRT"],
        "mos_results": [{"country": "ESP", "mos": None}, {"country": "PRT", "mos": None}],
        "attempts": 1,
    }

    allowed = nodes.allowed_actions(Situation.COLLECTION_FAILED, state)

    assert Action.CONTINUE_WITHOUT not in allowed
    assert Action.DELIVER_WITH_NOTE not in allowed


def _settings(**overrides):  # noqa: ANN202
    from app.config import Settings

    return Settings(**overrides)


async def test_the_model_is_asked_for_one_allowed_action_as_structured_output(monkeypatch) -> None:
    """The SDK call itself, with the client faked: the enum offered is exactly
    the fence, the answer is parsed from the JSON it must return, and its
    tokens join the run's tally."""
    import sys
    from types import SimpleNamespace

    from app import spend

    create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"action": "wait_and_resume", "reason": "temporary outage"}')],
            usage=SimpleNamespace(input_tokens=300, output_tokens=20),
        )
    )
    fake_sdk = SimpleNamespace(AsyncAnthropic=lambda **_kw: SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "anthropic", fake_sdk)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    with spend.tally() as spent:
        decision = await agent.decide(
            Situation.SOURCE_DOWN, {"missing_countries": ["PRT"]}, [Action.WAIT_AND_RESUME, Action.GIVE_UP]
        )

    assert decision == agent.Decision(Action.WAIT_AND_RESUME, "temporary outage", "model")
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5-5"
    schema = kwargs["output_config"]["format"]["schema"]
    assert schema["properties"]["action"]["enum"] == ["wait_and_resume", "give_up"]
    assert spent.input_tokens == 300 and spent.calls == 1
