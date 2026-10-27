from typing import Optional
from urllib.parse import quote

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Reads .env, the same way every other service here does. Without it this
    # service only ever saw real environment variables, so a .env sitting next
    # to it was silently ignored -- and the failure looked like a wrong token
    # rather than a file nobody read.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Path prefix a reverse proxy serves this API under and strips before
    # forwarding — "/orchestrator" on Azure, empty locally. Without it FastAPI
    # believes it sits at the root and builds the Swagger page pointing at an
    # absolute /openapi.json, which leaves this route entirely and 404s on
    # whichever ingress rule owns "/". Routing is unaffected; only the URLs the
    # docs page generates.
    root_path: str = ""
    log_level: str = "INFO"

    # The scoring service, and the shared secret its /internal tools require.
    # It is the ONLY service this one talks to: the collectors and the model
    # gateway sit behind its /internal tools, so their addresses are its
    # configuration, not this one's.
    # It is still deployed under the name marketmind-core; the DNS name changes
    # when the service is renamed, and this line is the only thing that has to
    # change with it.
    scoring_api_url: str = "http://marketmind-core:8000"
    scoring_internal_token: Optional[str] = None

    # How long to wait on a scoring call that collects. The scoring service
    # bounds one collection gap at about 1952 s (1800 dispatch + 120 polling +
    # the final poll) and runs a country's gaps in sequence within one call, so
    # no fixed figure is above every such call. A run waits as long as its own
    # deadline allows, never less than the floor — one gap plus margin, so a
    # run close to its deadline still lets the gap in flight finish. The max is
    # what a call made outside a run waits.
    scoring_collection_timeout_floor_seconds: float = 2100.0
    scoring_collection_timeout_max_seconds: float = 24 * 3600.0

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

    # --- The agent's decisions ---
    #
    # When a source fails, stays down, or a result does not hold together, a
    # model chooses what the run does next from a closed set of actions (see
    # app/agent.py). An unset key means the fixed rule decides, as it did
    # before the model existed.
    anthropic_api_key: Optional[str] = None
    decision_model: str = "claude-sonnet-5-5"
    # How long one "wait for the source to come back" lasts, at most. Never
    # past the run's deadline.
    source_wait_seconds: int = 30 * 60

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
