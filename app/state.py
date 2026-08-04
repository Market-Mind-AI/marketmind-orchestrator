from typing import TypedDict, Annotated, List
import operator

class OrchestratorState(TypedDict):
    analysis_id: str
    product_image_url: str
    product_description: str
    attributes: dict
    enrichment_results: Annotated[List[dict], operator.add]
    mos_score: dict
    report: dict
    errors: Annotated[List[str], operator.add]
