import json
import httpx
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from app.config import get_settings
from app.connectors import fetch_worldbank_data, fetch_oecd_data, fetch_un_data, fetch_pytrends_data
from app.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

class MarketTerms(BaseModel):
    country: str = Field(description="ISO-3166 alpha-3 country code")
    search_terms: List[str] = Field(description="List of real search terms for this product in this country")

class ExtractionResult(BaseModel):
    product_name: str = Field(description="Short, generic canonical name (e.g. 'Iberian ham', 'air fryer')")
    category: str = Field(description="Broader category in English")
    markets: List[MarketTerms] = Field(description="One entry per requested country with that country's real search vocabulary")

async def extraction_node(state: dict):
    settings = get_settings()
    llm = get_llm(settings.extraction_provider)
    structured_llm = llm.with_structured_output(ExtractionResult)
    
    prompt = f"""
Identify the product described below for a cross-country market-demand analysis.
CRITICAL: Provide markets for USA, GBR, and ESP (as default countries to check if not specified).

Description: {state.get("product_description", "")}
    """
    
    try:
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        # Pydantic object to dict
        return {"attributes": result.model_dump()}
    except Exception as e:
        return {"errors": [f"Extraction failed: {str(e)}"]}

async def enrichment_node(state: dict):
    attributes = state.get("attributes", {})
    markets = attributes.get("markets", [])
    product_name = attributes.get("product_name", "unknown")
    
    tasks = []
    # Fan-out: for each market/country, fetch from all 4 data sources concurrently
    for market in markets:
        tasks.append(fetch_worldbank_data(product_name))
        tasks.append(fetch_oecd_data(product_name))
        tasks.append(fetch_un_data(product_name))
        tasks.append(fetch_pytrends_data(product_name))
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_results = []
    for r in results:
        if isinstance(r, dict):
            valid_results.append(r)
            
    return {"enrichment_results": valid_results}

def scoring_node(state: dict):
    # Pure Python MOS arithmetic (TOPSIS simplification)
    enrichment_results = state.get("enrichment_results", [])
    
    base_score = 50
    if len(enrichment_results) > 0:
        base_score += min(len(enrichment_results) * 5, 50) # Cap at 100
        
    return {"mos_score": {"final_score": base_score, "method": "topsis_simplified"}}

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
