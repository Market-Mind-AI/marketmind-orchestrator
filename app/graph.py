from typing import Any

from langgraph.graph import END, StateGraph

from app.nodes import enrichment_node, extraction_node, report_node, scoring_node
from app.state import OrchestratorState


def build_graph(checkpointer: Any | None = None):  # noqa: ANN201
    """Compile the pipeline.

    Pass a checkpointer and every super-step is recorded under the run's
    thread_id, so re-invoking with that id resumes rather than restarting.
    Without one the graph still runs, and a restart loses the run -- which is
    why the application never builds it that way on purpose.
    """
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("extract", extraction_node)
    workflow.add_node("enrich", enrichment_node)
    workflow.add_node("score", scoring_node)
    workflow.add_node("report_generation", report_node)

    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "enrich")
    workflow.add_edge("enrich", "score")
    workflow.add_edge("score", "report_generation")
    workflow.add_edge("report_generation", END)

    return workflow.compile(checkpointer=checkpointer)
