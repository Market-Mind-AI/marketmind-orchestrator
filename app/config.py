from typing import Optional
from urllib.parse import quote

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Path prefix a reverse proxy serves this API under and strips before
    # forwarding — "/orchestrator" on Azure, empty locally. Without it FastAPI
    # believes it sits at the root and builds the Swagger page pointing at an
    # absolute /openapi.json, which leaves this route entirely and 404s on
    # whichever ingress rule owns "/". Routing is unaffected; only the URLs the
    # docs page generates.
    root_path: str = ""

    worldbank_api_url: str = "http://marketmind-worldbank:8000"
    oecd_api_url: str = "http://marketmind-oecd:8000"
    un_api_url: str = "http://marketmind-united-nations-data:8000"
    pytrends_api_url: str = "http://marketmind-pytrends:8000"
    llm_api_url: str = "http://marketmind-ollma-llm:8000"

    # The scoring service, and the shared secret its /internal tools require.
    # It is still deployed under the name marketmind-core; the DNS name changes
    # when the service is renamed, and this line is the only thing that has to
    # change with it.
    scoring_api_url: str = "http://marketmind-core:8000"
    scoring_internal_token: Optional[str] = None

    # Provider routing: "claude", "gpt", or "ollama"
    extraction_provider: str = "claude"
    report_provider: str = "claude"

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # --- Durable execution ---
    #
    # An analysis can run for hours: it insists on a source that is down rather
    # than dropping a market, and it may be resumed after the pod that started
    # it is gone. Where the graph got to has to outlive the process, so a
    # restart resumes instead of starting over and charging twice.
    #
    # "postgres" needs the connection below. "memory" keeps it in the process
    # and loses everything on restart, which is fine for a test and never fine
    # anywhere else — so it has to be asked for by name. Missing connection
    # details are a startup failure rather than a silent downgrade: durability
    # that quietly is not there is worse than none, because nothing looks wrong.
    checkpointer: str = "postgres"

    # Given in parts, not as a URL, matching every other service here. A URL in
    # the vault would carry a copy of the database password, and the password
    # already lives there once — changing it would then mean remembering two
    # places, with the second failing silently.
    #
    # checkpointer_database_url still wins when set, for a local one-liner.
    checkpointer_db_host: Optional[str] = None
    checkpointer_db_port: int = 5432
    checkpointer_db_name: Optional[str] = None
    checkpointer_db_user: Optional[str] = None
    checkpointer_db_password: Optional[str] = None
    checkpointer_database_url: Optional[str] = None

    @property
    def resolved_checkpointer_url(self) -> Optional[str]:
        if self.checkpointer_database_url:
            return self.checkpointer_database_url
        if not (self.checkpointer_db_host and self.checkpointer_db_name and self.checkpointer_db_user):
            return None
        password = quote(self.checkpointer_db_password or "", safe="")
        return (
            f"postgresql://{self.checkpointer_db_user}:{password}"
            f"@{self.checkpointer_db_host}:{self.checkpointer_db_port}/{self.checkpointer_db_name}"
        )

    # --- Observability ---
    #
    # Every run becomes a trace: the nodes, the tools called, the tokens, the
    # latency of each step. This is what answers "why did it do that" once the
    # decision stops being an `if` and starts being a model.
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"


def get_settings():
    return Settings()
