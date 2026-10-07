"""Time handling: day offsets, midnight crossing and duration arithmetic."""

from __future__ import annotations

from datetime import UTC, date, time, timedelta

import pytest

from engine.errors import DomainValidationError, InvalidJourneyError
from engine.time_utils import (
    add_days,
    anchor_date_for_boarding,
    boarding_anchor_date,
    combine,
    duration_minutes,
    first_departure_on_or_after,
    format_clock,
    minutes_between,
    minutes_from_midnight,
    parse_clock,
    stop_datetime,
)

TRAVEL_DATE = date(2026, 6, 15)


class TestClockParsing:
    def test_parses_hh_mm(self) -> None:
        assert parse_clock("23:30") == time(23, 30)

    def test_parses_midnight(self) -> None:
        assert parse_clock("00:00") == time(0, 0)

    def test_parses_hh_mm_ss_with_zero_seconds(self) -> None:
        assert parse_clock("05:40:00") == time(5, 40)

    def test_tolerates_surrounding_whitespace(self) -> None:
        assert parse_clock("  08:15 ") == time(8, 15)

    @pytest.mark.parametrize("text", ["", "  ", "8pm", "24:00", "23:60", "-1:00", "23", "abc"])
    def test_rejects_invalid_text(self, text: str) -> None:
        with pytest.raises(DomainValidationError):
            parse_clock(text)

    def test_rejects_sub_minute_precision(self) -> None:
        with pytest.raises(DomainValidationError):
            parse_clock("23:30:45")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(DomainValidationError):
            parse_clock(2330)  # type: ignore[arg-type]

    def test_formats_with_zero_padding(self) -> None:
        assert format_clock(time(5, 4)) == "05:04"
        assert format_clock(time(0, 0)) == "00:00"

    def test_minutes_from_midnight(self) -> None:
        assert minutes_from_midnight(time(0, 0)) == 0
        assert minutes_from_midnight(time(23, 30)) == 1410


class TestStopDatetimes:
    def test_day_offset_zero_is_the_anchor_date(self) -> None:
        assert stop_datetime(TRAVEL_DATE, 0, time(9, 15)) == combine(TRAVEL_DATE, time(9, 15))

    def test_day_offset_one_rolls_to_the_next_day(self) -> None:
        assert stop_datetime(TRAVEL_DATE, 1, time(5, 40)) == combine(
            TRAVEL_DATE + timedelta(days=1), time(5, 40)
        )

    def test_day_offset_two_rolls_two_days(self) -> None:
        assert stop_datetime(TRAVEL_DATE, 2, time(23, 0)).date() == TRAVEL_DATE + timedelta(days=2)

    def test_rejects_negative_day_offset(self) -> None:
        with pytest.raises(DomainValidationError):
            stop_datetime(TRAVEL_DATE, -1, time(1, 0))

    def test_rejects_datetime_supplied_as_date(self) -> None:
        with pytest.raises(DomainValidationError):
            stop_datetime(combine(TRAVEL_DATE, time(1, 0)), 0, time(1, 0))  # type: ignore[arg-type]


class TestMidnightCrossing:
    """The headline correctness case: BLR 23:30 (day 0) -> HYD 05:40 (day 1)."""

    def test_overnight_duration_is_positive_and_exact(self) -> None:
        departure = stop_datetime(TRAVEL_DATE, 0, time(23, 30))
        arrival = stop_datetime(TRAVEL_DATE, 1, time(5, 40))
        assert duration_minutes(departure, arrival) == 370

    def test_clock_only_comparison_would_be_wrong(self) -> None:
        """Guards against the classic bug of comparing bare clock times."""
        departure = stop_datetime(TRAVEL_DATE, 0, time(23, 30))
        arrival = stop_datetime(TRAVEL_DATE, 1, time(5, 40))
        assert arrival > departure
        assert minutes_between(departure, arrival) > 0

    def test_same_clock_time_next_day_is_24_hours(self) -> None:
        departure = stop_datetime(TRAVEL_DATE, 0, time(6, 0))
        arrival = stop_datetime(TRAVEL_DATE, 1, time(6, 0))
        assert duration_minutes(departure, arrival) == 1440

    def test_multi_day_route_accumulates_offsets(self) -> None:
        departure = stop_datetime(TRAVEL_DATE, 0, time(21, 0))
        arrival = stop_datetime(TRAVEL_DATE, 2, time(23, 0))
        assert duration_minutes(departure, arrival) == 50 * 60

    def test_add_days_crosses_month_boundary(self) -> None:
        assert add_days(date(2026, 6, 30), 1) == date(2026, 7, 1)


class TestBoardingAnchor:
    def test_anchor_shifts_back_for_boarding_day_offset(self) -> None:
        """A stop with day offset 1 boards the day after its anchor."""
        threshold = combine(TRAVEL_DATE, time(0, 0))
        anchor = boarding_anchor_date(threshold, time(5, 40), 1)
        assert anchor == TRAVEL_DATE - timedelta(days=1)
        assert stop_datetime(anchor, 1, time(5, 40)) == combine(TRAVEL_DATE, time(5, 40))

    def test_anchor_is_threshold_date_when_offset_is_zero(self) -> None:
        threshold = combine(TRAVEL_DATE, time(0, 0))
        assert boarding_anchor_date(threshold, time(23, 30), 0) == TRAVEL_DATE

    def test_boarding_anchor_for_first_segment_is_the_travel_date(self) -> None:
        assert anchor_date_for_boarding(TRAVEL_DATE, 0) == TRAVEL_DATE
        assert anchor_date_for_boarding(TRAVEL_DATE, 2) == TRAVEL_DATE - timedelta(days=2)

    def test_first_departure_rolls_to_next_day_when_time_already_passed(self) -> None:
        threshold = combine(TRAVEL_DATE, time(20, 0))
        assert first_departure_on_or_after(threshold, time(14, 30)) == combine(
            TRAVEL_DATE + timedelta(days=1), time(14, 30)
        )

    def test_first_departure_is_same_day_when_time_is_still_ahead(self) -> None:
        threshold = combine(TRAVEL_DATE, time(6, 0))
        assert first_departure_on_or_after(threshold, time(14, 30)) == combine(
            TRAVEL_DATE, time(14, 30)
        )

    def test_first_departure_at_exactly_the_threshold_is_kept(self) -> None:
        threshold = combine(TRAVEL_DATE, time(14, 30))
        assert first_departure_on_or_after(threshold, time(14, 30)) == threshold


class TestDurations:
    def test_exact_minutes(self) -> None:
        start = combine(TRAVEL_DATE, time(14, 0))
        end = combine(TRAVEL_DATE, time(23, 31))
        assert minutes_between(start, end) == 571

    def test_zero_length_is_rejected(self) -> None:
        moment = combine(TRAVEL_DATE, time(10, 0))
        with pytest.raises(InvalidJourneyError):
            duration_minutes(moment, moment)

    def test_backwards_time_is_rejected(self) -> None:
        start = combine(TRAVEL_DATE, time(10, 0))
        end = combine(TRAVEL_DATE, time(9, 0))
        with pytest.raises(InvalidJourneyError):
            duration_minutes(start, end)

    def test_duration_is_never_negative(self) -> None:
        start = combine(TRAVEL_DATE, time(23, 59))
        end = combine(TRAVEL_DATE + timedelta(days=1), time(0, 1))
        assert duration_minutes(start, end) == 2

    def test_rejects_timezone_aware_datetimes(self) -> None:
        from datetime import datetime

        aware = datetime(2026, 6, 15, 10, 0, tzinfo=UTC)
        with pytest.raises(DomainValidationError):
            minutes_between(aware, aware)
