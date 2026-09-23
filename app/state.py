import operator
from typing import Annotated, Any, TypedDict


class OrchestratorState(TypedDict, total=False):
    """What a run carries between steps, and what survives a restart.

    REFERENCES, not data. The identifiers, which countries are done, which
    sources were tried, the ruler in force. The indicator values, the scores and
    the evidence packet live in the scoring service: putting them here too would
    write megabytes into every checkpoint and leave two versions of the truth
    about one analysis.

    `mos_results` and `breakdown` are the exception, and only at the end. They
    are what gets delivered, and the delivery has to survive a restart between
    computing them and recording them, or the buyer pays for work that was done
    and thrown away.
    """

    # The run's identity. The checkpoint is keyed by this, which is what makes
    # re-invoking mean "resume" rather than "start another one".
    analysis_id: str

    entity_key: str | None
    canonical_title: str | None
    generic_entity_key: str | None
    countries: list[str]
    user_prompt: str | None
    product_description: str
    product_image_url: str

    # Decided ONCE, at the start, and carried. An analysis that begins before
    # midnight and resumes after it must not silently change which month it is
    # measuring, and a step that reads the clock cannot be replayed to the same
    # answer.
    period_key: str

    # When the buyer's wait began. Set once and never refreshed on resume: the
    # latency that matters is the one that was waited, and a run that died and
    # came back was waited through, restart included.
    started_at: str

    # How long to keep insisting on a source that is down before declaring the
    # analysis failed. Absent means the run has not started insisting yet.
    deadline_at: str | None

    # What each country's collection attempt produced. The tasks, not the data:
    # a FAILED task is a source that is down and worth retrying, while a task
    # that finished without failing and still left an indicator missing is a
    # source that does not hold that datum. Insisting on the second never ends.
    coverage_tasks: Annotated[list[dict[str, Any]], operator.add]
    countries_collected: Annotated[list[str], operator.add]

    # Whether the period's champion was already known. Cached means the slowest
    # collection in the system can be skipped, and skipping it is most of what
    # makes a repeated analysis fast.
    champion_cached: bool
    # The champion already stored for this period, when there is one. Carried
    # because the comparison needs the figure: a stored champion a requested
    # market now beats has to be replaced rather than crowned again.
    cached_champion_country: str | None
    cached_champion_mos: float | None
    champion_country: str | None

    # The ruler the countries' demand is read against, and which ruler it is.
    # A normalised figure presented as an absolute one is a false claim, so the
    # narrative has to be able to say which one it read.
    demand_ceiling: float | None
    demand_ruler: dict[str, Any] | None

    mos_results: list[dict[str, Any]]
    breakdown: dict[str, Any] | None
    narrative: str | None

    # Requested countries answered from long-term memory this round, mapped to
    # when that memory was recorded -- not accumulated across rounds, because
    # collection re-checks every requested country each time it runs, so this
    # is recomputed fresh and is always the current, complete picture. Carried
    # so give_up can name the date instead of repeating "no data" as if it had
    # just been discovered, and so the run can report how many markets its own
    # memory answered instead of collecting.
    recalled_countries: dict[str, str]

    # How many times the run has gone back to collect a country that is still
    # incomplete. Bounded, because a source that is permanently missing a datum
    # would otherwise be asked forever.
    attempts: int

    errors: Annotated[list[str], operator.add]
