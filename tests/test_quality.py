"""The three measures that need a definition before they can be a number."""

from app import quality

_PACKET = {
    "countries": [
        {"country": "ESP", "mos": 40.0219231872243, "signals": [{"score": 69.03, "context": "1500000 searches"}]},
        {"country": "PRT", "mos": 36.43655591042611},
    ]
}


# --- every figure the narrative states has to be in the evidence ---


def test_a_rounded_figure_is_supported() -> None:
    """Rounding is how the text is meant to read: a packet holding 40.0219
    supports "40"."""
    assert quality.unsupported_numbers("Spain scores 40", _PACKET) == []


def test_a_figure_nobody_computed_is_caught() -> None:
    assert quality.unsupported_numbers("Spain scores 55", _PACKET) == [55.0]


def test_a_fabricated_volume_is_caught() -> None:
    """The damaging kind: an absolute-sounding number that reads as measured."""
    assert quality.unsupported_numbers("about 9,400,000 searches a month", _PACKET) == [9400000.0]


def test_a_thousands_separated_figure_is_read_not_dropped() -> None:
    """Both spellings. A parser that silently skips what it cannot read
    approves everything it cannot read, which is worse than not checking."""
    assert quality.numbers_in("1.500.000") == [1500000.0]
    assert quality.numbers_in("1,500,000") == [1500000.0]
    assert quality.unsupported_numbers("1.500.000 searches", _PACKET) == []


def test_years_and_small_counts_are_not_claims_about_the_data() -> None:
    """"over the last 3 years" and "in 2026" are prose. Flagging them would
    bury the real finding under noise."""
    assert quality.unsupported_numbers("over the last 3 years, through 2026", _PACKET) == []


def test_an_empty_packet_accuses_nothing() -> None:
    """No evidence is not evidence of fabrication — it is a missing packet, and
    saying every figure is unsupported there would be a false alarm."""
    assert quality.unsupported_numbers("Spain scores 55", {}) == []


# --- the language rules are rules, so no model is asked ---


def test_search_interest_may_not_be_called_proven_demand() -> None:
    violations = quality.language_violations("The data shows proven demand in Spain.")

    assert len(violations) == 1
    assert "not proven demand" in violations[0]


def test_the_score_may_not_be_called_a_guarantee() -> None:
    assert quality.language_violations("This guarantees a successful entry.")


def test_careful_wording_passes() -> None:
    assert quality.language_violations("Search interest in Spain is moderate and rising.") == []


# --- how well the run used its tools ---


def test_collecting_the_same_country_twice_is_counted() -> None:
    """Where the cost leaks."""
    faults = quality.tool_call_faults(
        [{"country": "ESP"}, {"country": "ESP"}, {"country": "PRT"}], ["ESP", "PRT"]
    )

    assert faults["redundant"] == 1
    assert faults["off_request"] == 0


def test_collecting_a_country_nobody_asked_about_is_counted() -> None:
    faults = quality.tool_call_faults([{"country": "JPN"}], ["ESP", "PRT"])

    assert faults["off_request"] == 1


def test_a_clean_run_counts_nothing() -> None:
    faults = quality.tool_call_faults([{"country": "ESP"}, {"country": "PRT"}], ["ESP", "PRT"])

    assert faults == {"redundant": 0, "off_request": 0}


def test_a_population_restated_in_millions_is_the_same_figure() -> None:
    """Real, from a Brazilian narrative: it wrote the exact 212.812.405 the
    packet holds and then restated it as "212,8 milhões". The checker read the
    second as 212.8, found nothing like it, and reported a hallucination — with
    an absolute tolerance of 0.5 against a number twelve thousand away, at a
    scale where twelve thousand is nothing.

    Counting ordinary writing as invention buries the findings that matter.
    """
    from app.quality import unsupported_numbers

    packet = {"countries": [{"country": "BRA", "population": 212_812_405}]}

    assert unsupported_numbers("Com 212,8 milhões de habitantes.", packet) == []
    assert unsupported_numbers("Com 212.812.405 habitantes.", packet) == []


def test_a_figure_at_the_wrong_magnitude_is_still_caught() -> None:
    """The allowance is for the same number written differently, not for any
    number within a factor of ten. 300 million is not 212.8 million."""
    from app.quality import unsupported_numbers

    packet = {"countries": [{"country": "BRA", "population": 212_812_405}]}

    assert unsupported_numbers("Com 300 milhões de habitantes.", packet) == [300.0]


def test_a_score_is_still_checked_tightly() -> None:
    """The proportional allowance must not loosen the scores, which are the
    claims that decide the answer. 40.02 supports "40"; it does not support 44."""
    from app.quality import unsupported_numbers

    packet = {"countries": [{"country": "ESP", "mos": 40.0219}]}

    assert unsupported_numbers("Spain scores 40.", packet) == []
    assert unsupported_numbers("Spain scores 44.", packet) == [44.0]
