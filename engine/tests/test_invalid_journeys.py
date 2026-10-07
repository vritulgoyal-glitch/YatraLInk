"""Invalid, impossible and malformed journeys must be rejected with a reason."""

from __future__ import annotations

from datetime import timedelta

import pytest

from engine.candidates import JourneyPlan
from engine.config import SearchConfiguration
from engine.enums import AvailabilityState, JourneyType, RejectionReason
from engine.errors import DomainValidationError, InvalidJourneyError, UnknownStationError
from engine.journey import try_build_journey_option, validate_journey_option
from engine.legs import RailLeg
from engine.models import JourneyOption
from engine.pipeline import generate_journeys
from engine.search import SearchRequest
from engine.tests.builders import TRAVEL_DATE, at, make_segment, make_train


@pytest.fixture()
def config() -> SearchConfiguration:
    return SearchConfiguration()


class TestImpossibleRequests:
    def test_same_origin_and_destination_is_rejected_before_generation(self) -> None:
        with pytest.raises(ValueError):
            SearchRequest(origin="BLR", destination="BLR", travel_date=TRAVEL_DATE)

    def test_unknown_origin_station_raises(self, timetable, availability_book, config) -> None:
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        object.__setattr__(request, "origin", "ZZZ")
        with pytest.raises(UnknownStationError):
            generate_journeys(
                request,
                timetable=timetable,
                availability_book=availability_book,
                configuration=config,
            )

    def test_reverse_route_is_not_invented(self, timetable, availability_book, config) -> None:
        """No train runs DEL -> BLR, and the engine never reverses a route."""
        request = SearchRequest(origin="DEL", destination="BLR", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.options == ()
        assert result.generated_plan_count == 0

    def test_more_than_two_train_changes_is_not_generated(
        self, timetable, availability_book, config
    ) -> None:
        """AAA -> EEE needs four train changes, beyond the Phase 1 limit."""
        request = SearchRequest(origin="AAA", destination="EEE", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.options == ()

    def test_no_journey_ever_exceeds_the_change_limit(
        self, timetable, availability_book, config
    ) -> None:
        for origin, destination in [
            ("BLR", "DEL"),
            ("BLR", "BPL"),
            ("HYD", "DEL"),
            ("AAA", "EEE"),
            ("BLR", "NDLS"),
        ]:
            request = SearchRequest(origin=origin, destination=destination, travel_date=TRAVEL_DATE)
            result = generate_journeys(
                request,
                timetable=timetable,
                availability_book=availability_book,
                configuration=config,
            )
            for journey in result.options:
                assert journey.train_changes <= 2
                assert len(journey.segments) <= 3

    def test_no_journey_contains_a_repeated_station(
        self, timetable, availability_book, config
    ) -> None:
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for journey in result.options:
            path = journey.station_path
            assert len(set(path)) == len(path), path


class TestImpossiblePlans:
    def test_plan_with_non_contiguous_stations_is_rejected(self, timetable) -> None:
        yt1002 = timetable.train("YT1002")  # DEL -> NGP -> HYD
        yt1001 = timetable.train("YT1001")  # BLR -> HYD -> DEL
        first = RailLeg(train=yt1002, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE)
        second = RailLeg(train=yt1001, origin_index=1, destination_index=2, anchor_date=TRAVEL_DATE)
        with pytest.raises(DomainValidationError):
            JourneyPlan(JourneyType.CONNECTING, (first, second))

    def test_connection_below_minimum_is_rejected_with_a_reason(
        self, timetable, availability_book, config
    ) -> None:
        first_train = make_train("YT9001", [("AAA", None, "06:00", 0), ("BBB", "10:00", None, 0)])
        second_train = make_train("YT9002", [("BBB", None, "10:20", 0), ("CCC", "20:00", None, 0)])
        plan = JourneyPlan(
            JourneyType.CONNECTING,
            (
                RailLeg(
                    train=first_train, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE
                ),
                RailLeg(
                    train=second_train, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE
                ),
            ),
        )
        request = SearchRequest(origin="AAA", destination="CCC", travel_date=TRAVEL_DATE)
        outcome = try_build_journey_option(
            plan,
            request=request,
            timetable=timetable,
            availability_book=availability_book,
            config=config,
        )
        assert outcome.accepted is False
        assert outcome.reason is RejectionReason.INVALID_CONNECTION

    def test_accepted_plan_reports_success(self, timetable, availability_book, config) -> None:
        first_train = make_train("YT9001", [("AAA", None, "06:00", 0), ("BBB", "10:00", None, 0)])
        second_train = make_train("YT9002", [("BBB", None, "11:00", 0), ("CCC", "20:00", None, 0)])
        plan = JourneyPlan(
            JourneyType.CONNECTING,
            (
                RailLeg(
                    train=first_train, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE
                ),
                RailLeg(
                    train=second_train, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE
                ),
            ),
        )
        request = SearchRequest(origin="AAA", destination="CCC", travel_date=TRAVEL_DATE)
        outcome = try_build_journey_option(
            plan,
            request=request,
            timetable=timetable,
            availability_book=availability_book,
            config=config,
        )
        assert outcome.accepted is True
        assert outcome.option is not None
        assert outcome.reason is None

    def test_not_available_segment_makes_the_journey_untravellable(
        self, timetable, availability_book, config
    ) -> None:
        """YT1004 HYD -> DEL is NOT_AVAILABLE, so that journey must be rejected."""
        request = SearchRequest(origin="HYD", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert all(journey.train_numbers != ("YT1004",) for journey in result.options)
        reasons = {rejection.reason for rejection in result.rejections}
        assert RejectionReason.SEGMENT_NOT_AVAILABLE in reasons

    def test_not_available_can_be_included_only_when_explicitly_configured(
        self, timetable, availability_book
    ) -> None:
        permissive = SearchConfiguration(include_not_available_journeys=True)
        request = SearchRequest(origin="HYD", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=permissive,
        )
        yt1004 = [journey for journey in result.options if journey.train_numbers == ("YT1004",)]
        assert yt1004
        assert yt1004[0].availability is AvailabilityState.NOT_AVAILABLE
        assert not yt1004[0].is_confirmed


class TestRequestConstraintEnforcement:
    def test_earliest_departure_filters_out_earlier_journeys(
        self, timetable, availability_book
    ) -> None:
        from datetime import time

        request = SearchRequest(
            origin="BLR",
            destination="DEL",
            travel_date=TRAVEL_DATE,
            earliest_departure=time(20, 0),
        )
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=SearchConfiguration(),
        )
        assert result.options
        for journey in result.options:
            assert journey.departure >= at(TRAVEL_DATE, 20, 0)

    def test_latest_arrival_filters_out_later_journeys(
        self, timetable, availability_book, config
    ) -> None:
        deadline = at(TRAVEL_DATE + timedelta(days=1), 23, 0)
        request = SearchRequest(
            origin="BLR", destination="DEL", travel_date=TRAVEL_DATE, latest_arrival=deadline
        )
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for journey in result.options:
            assert journey.arrival <= deadline

    def test_impossible_deadline_yields_no_journeys(
        self, timetable, availability_book, config
    ) -> None:
        """No journey can reach DEL by midday on the travel date itself."""
        request = SearchRequest(
            origin="BLR",
            destination="DEL",
            travel_date=TRAVEL_DATE,
            latest_arrival=at(TRAVEL_DATE, 12, 0),
        )
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.options == ()

    def test_latest_arrival_violation_is_rejected_by_validation(self, config) -> None:
        """The defensive journey-level check rejects a journey that misses the deadline."""
        from engine.tests.builders import make_journey

        train = make_train("YT9001", [("AAA", None, "08:00", 0), ("CCC", "18:00", None, 0)])
        option = make_journey(train, "AAA", "CCC")
        request = SearchRequest(
            origin="AAA",
            destination="CCC",
            travel_date=TRAVEL_DATE,
            latest_arrival=at(TRAVEL_DATE, 17, 0),
        )
        with pytest.raises(InvalidJourneyError) as error:
            validate_journey_option(option, request=request, config=config)
        assert error.value.reason is RejectionReason.AFTER_LATEST_ARRIVAL

    def test_request_change_limit_is_honoured(self, timetable, availability_book, config) -> None:
        request = SearchRequest(
            origin="BLR",
            destination="DEL",
            travel_date=TRAVEL_DATE,
            max_train_changes=0,
        )
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for journey in result.options:
            assert journey.train_changes == 0


class TestJourneyLevelValidation:
    def _two_segment_journey(self, first_fare: int | None = 100000) -> JourneyOption:
        first = make_train("YT9001", [("AAA", None, "06:00", 0), ("BBB", "10:00", None, 0)])
        second = make_train("YT9002", [("BBB", None, "11:00", 0), ("CCC", "20:00", None, 0)])
        return JourneyOption(
            journey_id="j",
            journey_type=JourneyType.CONNECTING,
            origin_station_code="AAA",
            destination_station_code="CCC",
            segments=(
                make_segment(first, "AAA", "BBB", fare_paise=first_fare),
                make_segment(second, "BBB", "CCC", reservation_index=1),
            ),
        )

    def test_wrong_origin_is_rejected(self, config) -> None:
        request = SearchRequest(origin="AAA", destination="CCC", travel_date=TRAVEL_DATE)
        option = self._two_segment_journey()
        object.__setattr__(option, "origin_station_code", "ZZZ")
        with pytest.raises(InvalidJourneyError) as error:
            validate_journey_option(option, request=request, config=config)
        assert error.value.reason is RejectionReason.MALFORMED

    def test_wrong_destination_is_rejected(self, config) -> None:
        request = SearchRequest(origin="AAA", destination="CCC", travel_date=TRAVEL_DATE)
        option = self._two_segment_journey()
        object.__setattr__(option, "destination_station_code", "ZZZ")
        with pytest.raises(InvalidJourneyError) as error:
            validate_journey_option(option, request=request, config=config)
        assert error.value.reason is RejectionReason.MALFORMED

    def test_too_many_train_changes_is_rejected(self) -> None:
        strict = SearchConfiguration(max_train_changes=0, max_segments=3)
        request = SearchRequest(origin="AAA", destination="CCC", travel_date=TRAVEL_DATE)
        with pytest.raises(InvalidJourneyError) as error:
            validate_journey_option(self._two_segment_journey(), request=request, config=strict)
        assert error.value.reason is RejectionReason.TOO_MANY_TRAIN_CHANGES

    def test_request_specific_change_limit_is_enforced(self, config) -> None:
        request = SearchRequest(
            origin="AAA", destination="CCC", travel_date=TRAVEL_DATE, max_train_changes=0
        )
        with pytest.raises(InvalidJourneyError) as error:
            validate_journey_option(self._two_segment_journey(), request=request, config=config)
        assert error.value.reason is RejectionReason.TOO_MANY_TRAIN_CHANGES

    def test_too_many_segments_is_rejected(self, timetable, availability_book, config) -> None:
        strict = SearchConfiguration(max_segments=2, max_train_changes=1)
        first = make_train("YT9001", [("AAA", None, "06:00", 0), ("BBB", "08:00", None, 0)])
        second = make_train("YT9002", [("BBB", None, "09:00", 0), ("CCC", "11:00", None, 0)])
        third = make_train("YT9003", [("CCC", None, "12:00", 0), ("DDD", "14:00", None, 0)])
        plan = JourneyPlan(
            JourneyType.CONNECTING,
            (
                RailLeg(train=first, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE),
                RailLeg(train=second, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE),
                RailLeg(train=third, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE),
            ),
        )
        request = SearchRequest(origin="AAA", destination="DDD", travel_date=TRAVEL_DATE)
        outcome = try_build_journey_option(
            plan,
            request=request,
            timetable=timetable,
            availability_book=availability_book,
            config=strict,
        )
        assert outcome.accepted is False
        assert outcome.reason is RejectionReason.TOO_MANY_SEGMENTS

    def test_valid_journey_passes_validation(self, config) -> None:
        request = SearchRequest(origin="AAA", destination="CCC", travel_date=TRAVEL_DATE)
        validate_journey_option(self._two_segment_journey(), request=request, config=config)

    def test_unknown_fare_does_not_block_validation(self, config) -> None:
        request = SearchRequest(origin="AAA", destination="CCC", travel_date=TRAVEL_DATE)
        validate_journey_option(
            self._two_segment_journey(first_fare=None), request=request, config=config
        )

    def test_malformed_candidate_is_rejected_before_ranking(
        self, timetable, availability_book, config
    ) -> None:
        """A candidate that cannot be built must never reach the ranked output."""
        request = SearchRequest(origin="AAA", destination="EEE", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for journey in result.options:
            assert journey.station_path[0] == "AAA"
            assert journey.station_path[-1] == "EEE"
            assert journey.total_duration_minutes > 0

    def test_circular_path_is_not_generated(self, timetable, availability_book, config) -> None:
        for origin, destination in [("BLR", "DEL"), ("BLR", "BPL"), ("HYD", "DEL")]:
            request = SearchRequest(origin=origin, destination=destination, travel_date=TRAVEL_DATE)
            result = generate_journeys(
                request,
                timetable=timetable,
                availability_book=availability_book,
                configuration=config,
            )
            for journey in result.options:
                path = journey.station_path
                assert path[0] == origin
                assert path[-1] == destination
                assert len(path) == len(set(path))

    def test_destination_is_never_used_as_a_transfer_point(
        self, timetable, availability_book, config
    ) -> None:
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for journey in result.options:
            assert journey.destination_station_code not in journey.station_path[:-1]

    def test_no_journey_departs_before_it_arrives(
        self, timetable, availability_book, config
    ) -> None:
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for journey in result.options:
            assert journey.arrival > journey.departure
            for segment in journey.segments:
                assert segment.arrival > segment.departure
