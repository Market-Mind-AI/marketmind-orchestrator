from langgraph.graph import StateGraph, END
from app.state import OrchestratorState
from app.nodes import extraction_node, enrichment_node, scoring_node, report_node

def build_graph():
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
    
    return workflow.compile()
