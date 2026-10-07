"""Direct journey generation, end to end against the synthetic fixture snapshot."""

from __future__ import annotations

import pytest

from engine.candidates import generate_direct_plans
from engine.enums import AvailabilityState, JourneyType
from engine.errors import UnknownStationError
from engine.legs import RailLeg, legs_from_station
from engine.money import Fare
from engine.pipeline import generate_journeys
from engine.search import SearchRequest
from engine.tests.builders import TRAVEL_DATE, journeys_of_type


@pytest.fixture()
def blr_del_request() -> SearchRequest:
    return SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)


class TestLegs:
    def test_leg_computes_absolute_overnight_times(self, timetable) -> None:
        train = timetable.train("YT1001")
        leg = RailLeg(train=train, origin_index=0, destination_index=2, anchor_date=TRAVEL_DATE)
        assert leg.departure.isoformat() == "2026-06-15T23:30:00"
        assert leg.arrival.isoformat() == "2026-06-16T22:00:00"
        assert leg.duration_minutes == 1350

    def test_leg_rejects_reversed_route_indices(self, timetable) -> None:
        train = timetable.train("YT1001")
        with pytest.raises(ValueError):
            RailLeg(train=train, origin_index=2, destination_index=0, anchor_date=TRAVEL_DATE)

    def test_leg_rejects_boarding_at_the_terminus(self, timetable) -> None:
        train = timetable.train("YT1001")
        with pytest.raises(ValueError):
            RailLeg(train=train, origin_index=2, destination_index=2, anchor_date=TRAVEL_DATE)

    def test_legs_from_station_are_deterministic(self, timetable, config) -> None:
        from datetime import time as clock

        from engine.time_utils import combine

        threshold = combine(TRAVEL_DATE, clock(0, 0))
        first = legs_from_station(timetable, "BLR", threshold, config=config)
        second = legs_from_station(timetable, "BLR", threshold, config=config)
        assert [leg.describe() for leg in first] == [leg.describe() for leg in second]
        assert first

    def test_legs_from_station_orders_trains_by_number(self, timetable, config) -> None:
        from datetime import time as clock

        from engine.time_utils import combine

        threshold = combine(TRAVEL_DATE, clock(0, 0))
        legs = legs_from_station(timetable, "BLR", threshold, config=config)
        numbers = [leg.train_number for leg in legs]
        assert numbers == sorted(numbers)

    def test_destination_filter_keeps_only_that_destination(self, timetable, config) -> None:
        from datetime import time as clock

        from engine.time_utils import combine

        threshold = combine(TRAVEL_DATE, clock(0, 0))
        legs = legs_from_station(
            timetable, "BLR", threshold, config=config, destination_filter="DEL"
        )
        assert legs
        assert {leg.destination_station_code for leg in legs} == {"DEL"}

    def test_excluded_trains_are_skipped(self, timetable, config) -> None:
        from datetime import time as clock

        from engine.time_utils import combine

        threshold = combine(TRAVEL_DATE, clock(0, 0))
        legs = legs_from_station(
            timetable, "BLR", threshold, config=config, excluded_trains=frozenset({"YT1001"})
        )
        assert "YT1001" not in {leg.train_number for leg in legs}


class TestDirectGeneration:
    def test_direct_plans_are_single_segment(self, timetable, blr_del_request, config) -> None:
        plans = generate_direct_plans(timetable, blr_del_request, config)
        assert plans
        assert all(len(plan.segments) == 1 for plan in plans)
        assert all(plan.journey_type is JourneyType.DIRECT for plan in plans)

    def test_direct_plans_only_use_trains_serving_origin_before_destination(
        self, timetable, blr_del_request, config
    ) -> None:
        for plan in generate_direct_plans(timetable, blr_del_request, config):
            train = timetable.train(plan.segments[0].train_number)
            assert train.serves_in_order("BLR", "DEL")

    def test_reverse_route_produces_no_direct_plan(self, timetable, config) -> None:
        reverse = SearchRequest(origin="DEL", destination="BLR", travel_date=TRAVEL_DATE)
        assert generate_direct_plans(timetable, reverse, config) == ()

    def test_direct_plan_is_generated_for_a_known_direct_train(
        self, timetable, blr_del_request, config
    ) -> None:
        numbers = {
            plan.segments[0].train_number
            for plan in generate_direct_plans(timetable, blr_del_request, config)
        }
        assert "YT1001" in numbers
        assert "YT1003" in numbers
        assert "YT1013" in numbers

    def test_direct_generation_can_be_disabled(self, timetable, blr_del_request) -> None:
        from engine.config import SearchConfiguration

        disabled = SearchConfiguration(enable_direct=False)
        assert generate_direct_plans(timetable, blr_del_request, disabled) == ()

    def test_unknown_station_raises(self, timetable) -> None:
        with pytest.raises(UnknownStationError):
            timetable.station("ZZZ")


class TestDirectJourneysEndToEnd:
    def test_direct_journey_is_generated(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        direct = journeys_of_type(result, JourneyType.DIRECT)
        assert direct

    def test_yt1001_direct_journey_totals(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = next(
            j
            for j in journeys_of_type(result, JourneyType.DIRECT)
            if j.train_numbers == ("YT1001",)
        )
        assert journey.departure.isoformat() == "2026-06-15T23:30:00"
        assert journey.arrival.isoformat() == "2026-06-16T22:00:00"
        assert journey.total_duration_minutes == 1350
        assert journey.total_fare == Fare.from_paise(245000)
        assert journey.train_changes == 0
        assert journey.reservation_count == 1
        assert not journey.requires_separate_reservations
        assert journey.connections == ()
        assert journey.availability is AvailabilityState.AVAILABLE
        assert journey.is_confirmed

    def test_direct_journey_segment_details(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = next(
            j
            for j in journeys_of_type(result, JourneyType.DIRECT)
            if j.train_numbers == ("YT1001",)
        )
        segment = journey.segments[0]
        assert segment.train_name == "Sample Coromandel Express"
        assert segment.origin_station_name == "Bengaluru Sample Junction"
        assert segment.destination_station_name == "Delhi Sample Junction"
        assert segment.departure.date().isoformat() == "2026-06-15"
        assert segment.arrival.date().isoformat() == "2026-06-16"
        assert segment.seats_available == 42

    def test_request_ending_at_a_station_no_train_reaches_directly(
        self, timetable, availability_book, config
    ) -> None:
        """BLR -> BPL has no direct train, so no DIRECT candidate is produced."""
        request = SearchRequest(origin="BLR", destination="BPL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert journeys_of_type(result, JourneyType.DIRECT) == ()

    def test_all_returned_direct_journeys_are_single_segment(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for journey in journeys_of_type(result, JourneyType.DIRECT):
            assert len(journey.segments) == 1
            assert journey.segments[0].origin_station_code == "BLR"
            assert journey.segments[0].destination_station_code == "DEL"

    def test_direct_journey_has_no_connection_risk_entry(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        from engine.enums import ConnectionRisk

        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for journey in journeys_of_type(result, JourneyType.DIRECT):
            assert journey.risk is ConnectionRisk.SAFE
