import pytest

from app.nodes import enrichment_node, extraction_node, scoring_node


def test_scoring_refuses_instead_of_inventing_a_score():
    """It used to return 50 + 5 x len(results) labelled "topsis_simplified"."""
    with pytest.raises(NotImplementedError):
        scoring_node({"enrichment_results": [{"data": 10}]})


@pytest.mark.asyncio
async def test_enrichment_refuses_instead_of_inventing_volumes():
    """It used to answer {"volume": 100} whenever the call failed."""
    with pytest.raises(NotImplementedError):
        await enrichment_node({"attributes": {"markets": [], "product_name": "x"}})


@pytest.mark.asyncio
async def test_extraction_without_countries_reports_instead_of_assuming_three():
    """It used to put USA/GBR/ESP in the prompt when the request named no country."""
    result = await extraction_node({"product_description": "air fryer"})
    assert result["errors"]
    assert "countries" in result["errors"][0]
    assert "attributes" not in result
