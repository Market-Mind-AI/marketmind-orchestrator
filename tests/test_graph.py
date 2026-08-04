import pytest
from app.graph import build_graph

@pytest.mark.asyncio
async def test_graph_execution():
    graph = build_graph()
    result = await graph.ainvoke({"analysis_id": "123", "errors": [], "enrichment_results": []})
    assert "mos_score" in result
