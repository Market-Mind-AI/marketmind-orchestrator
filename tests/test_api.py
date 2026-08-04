from fastapi.testclient import TestClient
from app.main import app

def test_run_analysis_endpoint():
    client = TestClient(app)
    response = client.post("/run-analysis", json={"analysis_id": "123", "errors": [], "enrichment_results": []})
    assert response.status_code == 200
