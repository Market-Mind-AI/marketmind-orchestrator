from typing import TypedDict, Annotated, List
import operator

class OrchestratorState(TypedDict):
    analysis_id: str
    # The markets the caller asked about. Without them extraction has nothing to
    # scope itself to, and the node used to fill the gap with three hard-coded
    # countries.
    countries: List[str]
    product_image_url: str
    product_description: str
    attributes: dict
    enrichment_results: Annotated[List[dict], operator.add]
    mos_score: dict
    report: dict
    errors: Annotated[List[str], operator.add]
