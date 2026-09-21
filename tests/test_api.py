import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_run_analysis_streams_the_failure_instead_of_closing_quietly():
    response = client.post(
        "/run-analysis",
        json={"analysis_id": "123", "countries": [], "errors": [], "enrichment_results": []},
    )
    assert response.status_code == 200

    events = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert any("error" in e for e in events), "a failed run must say so on the stream"
    assert not any("mos_score" in json.dumps(e) for e in events), "no score may reach the caller"
