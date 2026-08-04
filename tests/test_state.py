def test_state_schema():
    from app.state import OrchestratorState
    state: OrchestratorState = {"analysis_id": "123", "errors": []}
    assert state["analysis_id"] == "123"
