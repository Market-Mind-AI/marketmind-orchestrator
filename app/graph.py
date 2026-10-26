from typing import Any

from langgraph.graph import END, StateGraph

from app.nodes import (
    after_check,
    after_decision,
    after_scoring,
    after_step,
    champion_node,
    check_result,
    collect_node,
    decide_node,
    deliver_node,
    enrich_node,
    evidence_node,
    give_up_node,
    measure_node,
    narrate_node,
    ruler_node,
    score_node,
    start_node,
    wait_node,
)
from app.state import OrchestratorState


def build_graph(checkpointer: Any | None = None):  # noqa: ANN201
    """The analysis, in the order the scoring service's own loop ran it.

    Linear while things go well. When they do not — a country whose collection
    failed after the scoring service's retries, a source or the scoring service
    itself down, a result that does not hold together before delivery — the
    run goes to `decide`, where the agent chooses from a closed set of actions
    (app/agent.py) and the edges carry the choice out through the same nodes:
    back to collect, a wait first, on to crowning and delivering with a note,
    one recompute, or giving up with a reason. A market the sources simply do
    not hold is not a decision; it ends the run with a reason, as before.

    Pass a checkpointer and every super-step is recorded under the run's
    thread_id, so re-invoking resumes rather than restarting.
    """
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("start", start_node)
    workflow.add_node("enrich", enrich_node)
    workflow.add_node("collect", collect_node)
    workflow.add_node("ruler", ruler_node)
    workflow.add_node("measure", measure_node)
    workflow.add_node("score", score_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("wait", wait_node)
    workflow.add_node("champion", champion_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("check", check_result)
    workflow.add_node("deliver", deliver_node)
    workflow.add_node("narrate", narrate_node)
    workflow.add_node("give_up", give_up_node)

    workflow.set_entry_point("start")
    workflow.add_edge("start", "enrich")
    workflow.add_edge("enrich", "collect")
    workflow.add_edge("collect", "ruler")
    # A step the scoring service could not finish goes to the decision rather
    # than failing the run: it is the same kind of failure as a source that is
    # down during collection.
    workflow.add_conditional_edges("ruler", after_step("measure"), {"measure": "measure", "decide": "decide"})
    workflow.add_conditional_edges("measure", after_step("score"), {"score": "score", "decide": "decide"})
    workflow.add_conditional_edges(
        "score",
        after_scoring,
        {"complete": "champion", "decide": "decide", "give_up": "give_up"},
    )

    # Going back means collection, not measurement: a country short of data
    # measured again produces the same absence. A recompute goes back to the
    # ruler — the data is there, the reading of it is what did not hold.
    workflow.add_conditional_edges(
        "decide",
        after_decision,
        {
            "collect": "collect",
            "wait": "wait",
            "ruler": "ruler",
            "champion": "champion",
            "deliver": "deliver",
            "give_up": "give_up",
        },
    )
    workflow.add_edge("wait", "collect")

    workflow.add_edge("champion", "evidence")
    workflow.add_edge("evidence", "check")
    workflow.add_conditional_edges("check", after_check, {"deliver": "deliver", "decide": "decide"})
    workflow.add_edge("deliver", "narrate")
    workflow.add_edge("narrate", END)
    workflow.add_edge("give_up", END)

    return workflow.compile(checkpointer=checkpointer)


# The compiled graph, put here by the lifespan so a background run can reach it.
# A run started by POST /analyze outlives the request that asked for it, so it
# cannot read the graph off `request.app.state`.
_graph: Any | None = None


def set_graph(graph: Any) -> None:
    global _graph
    _graph = graph


def get_graph() -> Any:
    if _graph is None:
        raise RuntimeError("The graph is not built yet; the application has not finished starting.")
    return _graph
