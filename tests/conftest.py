"""Keep the developer's .env out of the tests.

The application reads .env, which is right. A test suite that reads it is not:
it would pass or fail depending on what happens to be configured on the machine
running it, and the tests that check what happens when something is NOT set
would quietly stop checking anything on any machine where it is.
"""

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def _ignore_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    config = dict(Settings.model_config)
    config["env_file"] = None
    monkeypatch.setattr(Settings, "model_config", config)


@pytest.fixture(autouse=True)
def _no_decision_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key in the shell would send every test's decisions to a real model.
    Tests that exercise the model put a fake one in its place."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
