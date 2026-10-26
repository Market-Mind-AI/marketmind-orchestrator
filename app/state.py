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
    # A marca desta análise, quando há. Ela decide QUAL vocabulário se mede —
    # "juan valdez café" e "café" são buscas diferentes no Google Ads — e é
    # metade do par que a narrativa final explica.
    brand_key: str | None
    brand_name: str | None
    countries: list[str]
    user_prompt: str | None
    # The language the narrative comes out in — the buyer's interface
    # language, read back from the record. None is the scoring service's
    # configured default.
    language: str | None

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

    # The champion as the scoring service crowned it: country, figure, whether
    # it was requested, its row when it was not, and the ruler it was measured
    # against. Read, never decided here.
    champion: dict[str, Any] | None

    # The ruler the countries' demand is read against, and which ruler it is.
    # A normalised figure presented as an absolute one is a false claim, so the
    # narrative has to be able to say which one it read.
    demand_ceiling: float | None
    demand_ruler: dict[str, Any] | None

    # The answer went out. Read when something fails later in the run, or in a
    # run that resumed this one: a delivered analysis is never failed after
    # the fact, because the buyer already has it and was billed for it.
    delivered: bool

    mos_results: list[dict[str, Any]]
    breakdown: dict[str, Any] | None
    narrative: str | None

    # Requested markets the scoring service reported as having had no demand
    # recently, mapped to when the demand source last answered -- recomputed on
    # every collection round, never accumulated. Carried so give_up can name
    # the date instead of repeating "no data" as if it had just been found.
    absent_countries: dict[str, str]

    # How many times the run has gone back to collect a country that is still
    # incomplete. Bounded, because a source that is permanently missing a datum
    # would otherwise be asked forever.
    attempts: int

    errors: Annotated[list[str], operator.add]
