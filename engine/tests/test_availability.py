"""Availability representation, aggregation, and the UNKNOWN safety rule."""

from __future__ import annotations

import pytest

from engine.availability import (
    Availability,
    AvailabilityBook,
    DateAgnosticAvailability,
    summarize_availability,
)
from engine.enums import AVAILABILITY_DESIRABILITY, AVAILABILITY_ORDER, AvailabilityState
from engine.errors import DomainValidationError
from engine.money import Fare
from engine.tests.builders import TRAVEL_DATE

ALL_STATES = (
    AvailabilityState.AVAILABLE,
    AvailabilityState.RAC,
    AvailabilityState.WAITLIST,
    AvailabilityState.UNKNOWN,
    AvailabilityState.NOT_AVAILABLE,
)


def _entry(
    state: AvailabilityState,
    *,
    train: str = "YT1001",
    origin: str = "BLR",
    destination: str = "DEL",
    fare_paise: int | None = 100000,
) -> Availability:
    return Availability(
        train_number=train,
        origin_station_code=origin,
        destination_station_code=destination,
        travel_date=TRAVEL_DATE,
        travel_class="3A",
        state=state,
        fare=Fare.from_paise(fare_paise) if fare_paise is not None else None,
    )


class TestAvailabilityState:
    def test_desirability_ladder_is_documented_order(self) -> None:
        assert AVAILABILITY_DESIRABILITY[AvailabilityState.AVAILABLE] == 0
        assert AVAILABILITY_DESIRABILITY[AvailabilityState.RAC] == 1
        assert AVAILABILITY_DESIRABILITY[AvailabilityState.WAITLIST] == 2
        assert AVAILABILITY_DESIRABILITY[AvailabilityState.UNKNOWN] == 3
        assert AVAILABILITY_DESIRABILITY[AvailabilityState.NOT_AVAILABLE] == 4

    def test_order_constant_matches_the_ladder(self) -> None:
        expected = sorted(AVAILABILITY_DESIRABILITY, key=AVAILABILITY_DESIRABILITY.__getitem__)
        assert AVAILABILITY_ORDER == tuple(expected)

    def test_only_available_is_confirmed(self) -> None:
        assert AvailabilityState.AVAILABLE.is_confirmed
        for state in ALL_STATES[1:]:
            assert not state.is_confirmed

    def test_unknown_is_not_known(self) -> None:
        assert not AvailabilityState.UNKNOWN.is_known
        assert AvailabilityState.AVAILABLE.is_known

    def test_bookable_states_exclude_unknown_and_unavailable(self) -> None:
        bookable = {state for state in ALL_STATES if state.is_bookable}
        assert bookable == {
            AvailabilityState.AVAILABLE,
            AvailabilityState.RAC,
            AvailabilityState.WAITLIST,
        }

    def test_unknown_is_never_equal_to_available(self) -> None:
        assert AvailabilityState.UNKNOWN is not AvailabilityState.AVAILABLE
        assert AvailabilityState.UNKNOWN.value != AvailabilityState.AVAILABLE.value


class TestAvailabilityModel:
    def test_valid_entry(self) -> None:
        entry = _entry(AvailabilityState.AVAILABLE)
        assert entry.state is AvailabilityState.AVAILABLE
        assert entry.fare == Fare.from_paise(100000)
        assert entry.quota == "GENERAL"

    def test_state_accepts_a_string(self) -> None:
        entry = Availability(
            train_number="YT1",
            origin_station_code="AAA",
            destination_station_code="BBB",
            travel_date=TRAVEL_DATE,
            travel_class="3A",
            state="waitlist",
        )
        assert entry.state is AvailabilityState.WAITLIST

    def test_invalid_state_string_is_rejected(self) -> None:
        with pytest.raises(DomainValidationError):
            Availability(
                train_number="YT1",
                origin_station_code="AAA",
                destination_station_code="BBB",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
                state="CONFIRMED",
            )

    def test_identical_origin_and_destination_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Availability(
                train_number="YT1",
                origin_station_code="AAA",
                destination_station_code="AAA",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
                state=AvailabilityState.AVAILABLE,
            )

    def test_float_fare_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Availability(
                train_number="YT1",
                origin_station_code="AAA",
                destination_station_code="BBB",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
                state=AvailabilityState.AVAILABLE,
                fare=1234.5,  # type: ignore[arg-type]
            )

    def test_negative_seats_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            Availability(
                train_number="YT1",
                origin_station_code="AAA",
                destination_station_code="BBB",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
                state=AvailabilityState.AVAILABLE,
                seats_available=-1,
            )

    def test_quota_is_upper_cased(self) -> None:
        entry = Availability(
            train_number="YT1",
            origin_station_code="AAA",
            destination_station_code="BBB",
            travel_date=TRAVEL_DATE,
            travel_class="3A",
            state=AvailabilityState.AVAILABLE,
            quota="tatkal",
        )
        assert entry.quota == "TATKAL"

    def test_entry_is_hashable_by_key(self) -> None:
        entry = _entry(AvailabilityState.AVAILABLE)
        assert entry.key[:2] == ("YT1001", "BLR")


class TestAggregation:
    def test_single_available_segment(self) -> None:
        summary = summarize_availability([AvailabilityState.AVAILABLE])
        assert summary.state is AvailabilityState.AVAILABLE
        assert summary.all_confirmed
        assert summary.segment_count == 1

    def test_single_rac_segment(self) -> None:
        summary = summarize_availability([AvailabilityState.RAC])
        assert summary.state is AvailabilityState.RAC
        assert not summary.all_confirmed

    def test_single_waitlist_segment(self) -> None:
        assert (
            summarize_availability([AvailabilityState.WAITLIST]).state is AvailabilityState.WAITLIST
        )

    def test_single_not_available_segment(self) -> None:
        summary = summarize_availability([AvailabilityState.NOT_AVAILABLE])
        assert summary.state is AvailabilityState.NOT_AVAILABLE
        assert summary.has_not_available

    def test_single_unknown_segment(self) -> None:
        summary = summarize_availability([AvailabilityState.UNKNOWN])
        assert summary.state is AvailabilityState.UNKNOWN
        assert summary.has_unknown
        assert not summary.all_confirmed

    def test_worst_state_wins(self) -> None:
        summary = summarize_availability(
            [AvailabilityState.AVAILABLE, AvailabilityState.RAC, AvailabilityState.AVAILABLE]
        )
        assert summary.state is AvailabilityState.RAC
        assert not summary.all_confirmed

    def test_waitlist_outranks_rac_as_worse(self) -> None:
        summary = summarize_availability([AvailabilityState.RAC, AvailabilityState.WAITLIST])
        assert summary.state is AvailabilityState.WAITLIST

    def test_unknown_outranks_waitlist_as_worse(self) -> None:
        """Unknown inventory is not actionable, so it is worse than a known waitlist."""
        summary = summarize_availability([AvailabilityState.WAITLIST, AvailabilityState.UNKNOWN])
        assert summary.state is AvailabilityState.UNKNOWN

    def test_unknown_and_not_available_aggregate_to_not_available(self) -> None:
        """The worst state wins: NOT_AVAILABLE is worse than UNKNOWN."""
        summary = summarize_availability(
            [AvailabilityState.UNKNOWN, AvailabilityState.NOT_AVAILABLE]
        )
        assert summary.state is AvailabilityState.NOT_AVAILABLE

    def test_unknown_is_never_treated_as_confirmed(self) -> None:
        for states in [
            [AvailabilityState.UNKNOWN],
            [AvailabilityState.AVAILABLE, AvailabilityState.UNKNOWN],
            [AvailabilityState.UNKNOWN, AvailabilityState.UNKNOWN],
        ]:
            summary = summarize_availability(states)
            assert not summary.all_confirmed
            assert not summary.state.is_confirmed

    def test_aggregation_is_order_independent(self) -> None:
        forward = summarize_availability(
            [AvailabilityState.AVAILABLE, AvailabilityState.RAC, AvailabilityState.UNKNOWN]
        )
        backward = summarize_availability(
            [AvailabilityState.UNKNOWN, AvailabilityState.RAC, AvailabilityState.AVAILABLE]
        )
        assert forward.state is backward.state
        assert dict(forward.counts) == dict(backward.counts)

    def test_counts_are_reported(self) -> None:
        summary = summarize_availability(
            [AvailabilityState.AVAILABLE, AvailabilityState.AVAILABLE, AvailabilityState.RAC]
        )
        assert summary.counts[AvailabilityState.AVAILABLE] == 2
        assert summary.counts[AvailabilityState.RAC] == 1
        assert summary.segment_count == 3

    def test_empty_input_is_unknown_and_not_confirmed(self) -> None:
        summary = summarize_availability([])
        assert summary.state is AvailabilityState.UNKNOWN
        assert not summary.all_confirmed
        assert summary.segment_count == 0

    def test_describe_is_readable(self) -> None:
        summary = summarize_availability([AvailabilityState.AVAILABLE, AvailabilityState.RAC])
        text = summary.describe()
        assert "RAC" in text
        assert "AVAILABLE" in text

    def test_all_available_is_the_only_confirmed_case(self) -> None:
        assert summarize_availability([AvailabilityState.AVAILABLE] * 3).all_confirmed
        for state in ALL_STATES[1:]:
            assert not summarize_availability([AvailabilityState.AVAILABLE, state]).all_confirmed


class TestAvailabilityBook:
    def test_lookup_returns_the_reported_record(self) -> None:
        book = AvailabilityBook([_entry(AvailabilityState.RAC)])
        found = book.lookup(
            train_number="yt1001",
            origin_station_code="blr",
            destination_station_code="del",
            travel_date=TRAVEL_DATE,
            travel_class="3A",
        )
        assert found is not None
        assert found.state is AvailabilityState.RAC

    def test_lookup_returns_none_when_not_reported(self) -> None:
        book = AvailabilityBook([_entry(AvailabilityState.AVAILABLE)])
        assert (
            book.lookup(
                train_number="YT9999",
                origin_station_code="BLR",
                destination_station_code="DEL",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
            )
            is None
        )

    def test_lookup_is_sensitive_to_class(self) -> None:
        book = AvailabilityBook([_entry(AvailabilityState.AVAILABLE)])
        assert (
            book.lookup(
                train_number="YT1001",
                origin_station_code="BLR",
                destination_station_code="DEL",
                travel_date=TRAVEL_DATE,
                travel_class="SL",
            )
            is None
        )

    def test_lookup_is_sensitive_to_quota(self) -> None:
        book = AvailabilityBook([_entry(AvailabilityState.AVAILABLE)])
        assert (
            book.lookup(
                train_number="YT1001",
                origin_station_code="BLR",
                destination_station_code="DEL",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
                quota="TATKAL",
            )
            is None
        )

    def test_len_and_entries(self) -> None:
        book = AvailabilityBook([_entry(AvailabilityState.AVAILABLE)])
        assert len(book) == 1
        assert len(book.entries) == 1

    def test_non_availability_entry_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            AvailabilityBook([{"train": "x"}])  # type: ignore[list-item]

    def test_date_agnostic_entry_carries_no_date(self) -> None:
        entry = DateAgnosticAvailability(
            train_number="YT1",
            origin_station_code="AAA",
            destination_station_code="BBB",
            travel_class="3A",
            state=AvailabilityState.UNKNOWN,
        )
        assert entry.to_json()["travel_date"] is None
        assert entry.state is AvailabilityState.UNKNOWN


class TestAvailabilityInJourneys:
    def test_journey_with_unknown_segment_is_not_confirmed(
        self, timetable, availability_book, config
    ) -> None:
        """YT1014 HYD -> NDLS has no availability record at all."""
        from engine.pipeline import generate_journeys
        from engine.search import SearchRequest

        request = SearchRequest(origin="HYD", destination="NDLS", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.options
        for journey in result.options:
            assert not journey.is_confirmed

    def test_unknown_segment_fare_is_none_not_zero(
        self, timetable, availability_book, config
    ) -> None:
        from engine.pipeline import generate_journeys
        from engine.search import SearchRequest

        request = SearchRequest(origin="HYD", destination="NDLS", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = next(j for j in result.options if j.train_numbers == ("YT1014",))
        assert journey.segments[0].availability is AvailabilityState.UNKNOWN
        assert journey.segments[0].fare is None
        assert journey.total_fare is None

    def test_availability_summary_is_exposed_on_every_journey(
        self, timetable, availability_book, config
    ) -> None:
        from engine.pipeline import generate_journeys
        from engine.search import SearchRequest

        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.options
        for journey in result.options:
            summary = journey.availability_summary
            assert summary is not None
            assert summary.segment_count == len(journey.segments)
            assert summary.state is journey.availability
