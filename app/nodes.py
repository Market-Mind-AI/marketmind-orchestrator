async def extraction_node(state: dict):
    # Stub: Call Haiku Vision LLM
    return {"attributes": {"category": "stub"}}

async def enrichment_worldbank_node(state: dict):
    # Call fetch_worldbank_data
    return {"enrichment_results": [{"source": "wb"}]}

def scoring_node(state: dict):
    # Pure Python MOS arithmetic
    return {"mos_score": {"final_score": 85}}

async def report_node(state: dict):
    # Stub: Call marketmind-ollma-llm
    return {"report": {"narrative": "Stub report"}}
