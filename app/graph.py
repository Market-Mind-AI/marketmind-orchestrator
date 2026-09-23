from typing import Any

from langgraph.graph import END, StateGraph

from app.nodes import (
    after_scoring,
    champion_node,
    collect_node,
    deliver_node,
    enrich_node,
    evidence_node,
    give_up_node,
    measure_node,
    narrate_node,
    ruler_node,
    score_node,
    start_node,
)
from app.state import OrchestratorState


def build_graph(checkpointer: Any | None = None, store: Any | None = None):  # noqa: ANN201
    """The analysis, in the order the scoring service's own loop ran it.

    Linear except for one branch, and that branch is the analysis's real
    decision: a run that scored every requested market goes on to deliver, and
    one that did not either goes back to collect or ends with a reason. Nothing
    partial is delivered -- a country missing from a ranking is not a smaller
    answer, it is a different one.

    Pass a checkpointer and every super-step is recorded under the run's
    thread_id, so re-invoking resumes rather than restarting. Pass a store and
    the nodes that declare one -- collect, score, give_up -- receive it; the
    rest are unaffected, because LangGraph injects it by the node's own
    signature, not by anything declared here.
    """
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("start", start_node)
    workflow.add_node("enrich", enrich_node)
    workflow.add_node("collect", collect_node)
    workflow.add_node("ruler", ruler_node)
    workflow.add_node("measure", measure_node)
    workflow.add_node("score", score_node)
    workflow.add_node("champion", champion_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("deliver", deliver_node)
    workflow.add_node("narrate", narrate_node)
    workflow.add_node("give_up", give_up_node)

    workflow.set_entry_point("start")
    workflow.add_edge("start", "enrich")
    workflow.add_edge("enrich", "collect")
    workflow.add_edge("collect", "ruler")
    workflow.add_edge("ruler", "measure")
    workflow.add_edge("measure", "score")

    # Insisting goes back to collection, not to measurement: the country is
    # short of data, and measuring the same absence again produces the same
    # absence.
    workflow.add_conditional_edges(
        "score",
        after_scoring,
        {"complete": "champion", "insist": "collect", "give_up": "give_up"},
    )

    workflow.add_edge("champion", "evidence")
    workflow.add_edge("evidence", "deliver")
    workflow.add_edge("deliver", "narrate")
    workflow.add_edge("narrate", END)
    workflow.add_edge("give_up", END)

    return workflow.compile(checkpointer=checkpointer, store=store)


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
