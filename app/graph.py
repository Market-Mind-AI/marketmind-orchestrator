from langgraph.graph import StateGraph, END
from app.state import OrchestratorState
from app.nodes import extraction_node, enrichment_worldbank_node, scoring_node, report_node

def build_graph():
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("extract", extraction_node)
    workflow.add_node("enrich_wb", enrichment_worldbank_node)
    workflow.add_node("score", scoring_node)
    workflow.add_node("report_generation", report_node)
    
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "enrich_wb")
    workflow.add_edge("enrich_wb", "score")
    workflow.add_edge("score", "report_generation")
    workflow.add_edge("report_generation", END)
    
    return workflow.compile()
