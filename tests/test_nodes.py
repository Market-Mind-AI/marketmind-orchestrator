import pytest
from app.nodes import scoring_node

def test_scoring_node():
    state = {"enrichment_results": [{"data": 10}]}
    result = scoring_node(state)
    assert "mos_score" in result
