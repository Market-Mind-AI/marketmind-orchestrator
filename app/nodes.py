from typing import List

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.llm_factory import get_llm


class MarketTerms(BaseModel):
    country: str = Field(description="ISO-3166 alpha-3 country code")
    search_terms: List[str] = Field(description="List of real search terms for this product in this country")


class ExtractionResult(BaseModel):
    product_name: str = Field(description="Short, generic canonical name (e.g. 'Iberian ham', 'air fryer')")
    category: str = Field(description="Broader category in English")
    markets: List[MarketTerms] = Field(description="One entry per requested country with that country's real search vocabulary")


async def extraction_node(state: dict):
    # The countries come from the request. An earlier version hard-coded
    # USA/GBR/ESP into the prompt, which meant a request for four other markets
    # silently got vocabulary for three it never asked about — a wrong answer
    # that looked like a right one.
    countries = state.get("countries") or []
    if not countries:
        return {"errors": ["Extraction skipped: the request carried no countries"]}

    settings = get_settings()
    llm = get_llm(settings.extraction_provider)
    structured_llm = llm.with_structured_output(ExtractionResult)

    prompt = f"""
Identify the product described below for a cross-country market-demand analysis.
Provide one markets entry for EACH of these countries, and no others: {", ".join(countries)}.

Description: {state.get("product_description", "")}
    """

    try:
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        return {"attributes": result.model_dump()}
    except Exception as e:
        return {"errors": [f"Extraction failed: {str(e)}"]}


async def enrichment_node(state: dict):
    # Deliberately unimplemented. What stood here called GET /api/data on the
    # four collection services — a route none of them has — and, when that
    # failed, returned {"volume": 100} and friends. Invented numbers that
    # reached the score and could not be told apart from measured ones.
    #
    # Collection is the collection services' own job now: each keeps its data
    # fresh on a daily schedule and exposes what it holds. This node is wired to
    # that in step 8 of the migration plan; until then it must not pretend.
    raise NotImplementedError(
        "Enrichment is not wired to the collection services yet (migration step 8). "
        "It previously fabricated volumes when the call failed, so it now refuses instead."
    )


def scoring_node(state: dict):
    # Deliberately unimplemented. What stood here computed 50 + 5 x (number of
    # results), capped at 100, and labelled it "topsis_simplified". It is not
    # TOPSIS, it does not read an indicator, and it does not know what country
    # it is scoring.
    #
    # The MOS is Cap. 5 and it stays in the scoring service, which owns the
    # weights, the rulers and the reference population. This node becomes a call
    # to it in step 8.
    raise NotImplementedError(
        "Scoring belongs to the scoring service (Cap. 5) and is called from there in migration step 8. "
        "The arithmetic that stood here was not TOPSIS."
    )


async def report_node(state: dict):
    settings = get_settings()
    llm = get_llm(settings.report_provider)
    score = state.get("mos_score", {}).get("final_score", 0)
    product_name = state.get("attributes", {}).get("product_name", "the product")

    prompt = f"Write a short 2-sentence market analysis narrative for '{product_name}' which scored {score}/100 in demand volume and availability."

    try:
        result = await llm.ainvoke([
            SystemMessage(content="You are an expert market analyst."),
            HumanMessage(content=prompt)
        ])
        return {"report": {"narrative": result.content}}
    except Exception as e:
        return {"errors": [f"Reporting failed: {str(e)}"]}
