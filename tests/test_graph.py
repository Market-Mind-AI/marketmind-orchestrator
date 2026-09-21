import pytest

from app.graph import build_graph


@pytest.mark.asyncio
async def test_graph_cannot_produce_a_score_yet():
    """The pipeline is not wired to the real services, and says so rather than
    running to completion on fabricated data (migration step 8)."""
    graph = build_graph()
    with pytest.raises(NotImplementedError):
        await graph.ainvoke(
            {"analysis_id": "123", "countries": [], "errors": [], "enrichment_results": []}
        )
