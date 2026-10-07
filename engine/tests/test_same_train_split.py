"""Same-train split generation: one physical train, several reservations."""

from __future__ import annotations

import pytest

from engine.candidates import JourneyPlan, generate_same_train_split_plans
from engine.enums import AvailabilityState, JourneyType
from engine.errors import DomainValidationError
from engine.money import Fare
from engine.pipeline import generate_journeys
from engine.search import SearchRequest
from engine.tests.builders import TRAVEL_DATE, find_journey, journeys_of_type


@pytest.fixture()
def blr_del_request() -> SearchRequest:
    return SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)


class TestSplitGeneration:
    def test_splits_are_generated(self, timetable, blr_del_request, config) -> None:
        plans = generate_same_train_split_plans(timetable, blr_del_request, config)
        assert plans
        assert all(plan.journey_type is JourneyType.SAME_TRAIN_SPLIT for plan in plans)

    def test_every_split_uses_one_train(self, timetable, blr_del_request, config) -> None:
        for plan in generate_same_train_split_plans(timetable, blr_del_request, config):
            assert len({segment.train_number for segment in plan.segments}) == 1

    def test_split_segments_are_contiguous_and_ordered(
        self, timetable, blr_del_request, config
    ) -> None:
        for plan in generate_same_train_split_plans(timetable, blr_del_request, config):
            for previous, following in zip(plan.segments, plan.segments[1:], strict=False):
                assert previous.destination_station_code == following.origin_station_code
                assert following.departure >= previous.arrival

    def test_split_for_a_three_stop_train_has_exactly_two_reservations(
        self, timetable, blr_del_request, config
    ) -> None:
        """YT1001 runs BLR -> HYD -> DEL: the only split point is HYD."""
        yt1001 = [
            plan
            for plan in generate_same_train_split_plans(timetable, blr_del_request, config)
            if plan.segments[0].train_number == "YT1001"
        ]
        assert [len(plan.segments) for plan in yt1001] == [2]
        assert yt1001[0].segments[0].destination_station_code == "HYD"

    def test_generation_is_deterministic(self, timetable, blr_del_request, config) -> None:
        first = generate_same_train_split_plans(timetable, blr_del_request, config)
        second = generate_same_train_split_plans(timetable, blr_del_request, config)
        assert [plan.describe() for plan in first] == [plan.describe() for plan in second]

    def test_split_generation_can_be_disabled(self, timetable, blr_del_request) -> None:
        from engine.config import SearchConfiguration

        disabled = SearchConfiguration(enable_same_train_split=False)
        assert generate_same_train_split_plans(timetable, blr_del_request, disabled) == ()

    def test_split_limit_of_two_halves_the_candidates(self, timetable, blr_del_request) -> None:
        from engine.config import SearchConfiguration

        limited = SearchConfiguration(same_train_split_max_segments=2)
        plans = generate_same_train_split_plans(timetable, blr_del_request, limited)
        assert plans
        assert all(len(plan.segments) == 2 for plan in plans)

    def test_no_split_for_a_two_stop_train(self, timetable, config) -> None:
        """YT1006 runs HYD -> DEL with no intermediate stop, so it cannot split."""
        request = SearchRequest(origin="HYD", destination="DEL", travel_date=TRAVEL_DATE)
        plans = generate_same_train_split_plans(timetable, request, config)
        assert all(plan.segments[0].train_number != "YT1006" for plan in plans)

    def test_reverse_request_cannot_produce_a_split(self, timetable, config) -> None:
        reverse = SearchRequest(origin="DEL", destination="BLR", travel_date=TRAVEL_DATE)
        assert generate_same_train_split_plans(timetable, reverse, config) == ()


class TestSplitEndToEnd:
    def test_split_journey_is_returned(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert journeys_of_type(result, JourneyType.SAME_TRAIN_SPLIT)

    def test_split_has_zero_train_changes_and_two_reservations(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = find_journey(
            result, train_numbers=("YT1001", "YT1001"), journey_type=JourneyType.SAME_TRAIN_SPLIT
        )
        assert journey is not None
        assert journey.train_changes == 0
        assert journey.reservation_count == 2
        assert journey.requires_separate_reservations

    def test_split_segments_use_the_same_train_and_share_the_transfer_station(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = find_journey(
            result, train_numbers=("YT1001", "YT1001"), journey_type=JourneyType.SAME_TRAIN_SPLIT
        )
        assert journey is not None
        first, second = journey.segments
        assert first.train_number == second.train_number == "YT1001"
        assert first.destination_station_code == second.origin_station_code == "HYD"
        assert first.arrival <= second.departure

    def test_split_availability_is_evaluated_per_segment(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        """YT1001 BLR-HYD is AVAILABLE while HYD-DEL is WAITLIST."""
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = find_journey(
            result, train_numbers=("YT1001", "YT1001"), journey_type=JourneyType.SAME_TRAIN_SPLIT
        )
        assert journey is not None
        assert journey.segments[0].availability is AvailabilityState.AVAILABLE
        assert journey.segments[1].availability is AvailabilityState.WAITLIST
        assert journey.availability is AvailabilityState.WAITLIST
        assert not journey.is_confirmed

    def test_split_fare_is_the_sum_of_segment_fares(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = find_journey(
            result, train_numbers=("YT1001", "YT1001"), journey_type=JourneyType.SAME_TRAIN_SPLIT
        )
        assert journey is not None
        assert journey.total_fare == Fare.from_paise(98000 + 147000)
        assert journey.segments[0].fare == Fare.from_paise(98000)
        assert journey.segments[1].fare == Fare.from_paise(147000)

    def test_split_elapsed_duration_equals_the_whole_ride(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = find_journey(
            result, train_numbers=("YT1001", "YT1001"), journey_type=JourneyType.SAME_TRAIN_SPLIT
        )
        assert journey is not None
        assert journey.total_duration_minutes == 1350
        # 1350 minutes elapsed = 1345 on board + the train's own 5-minute halt.
        assert journey.total_travel_minutes == 1345
        assert journey.total_transfer_minutes == 5
        assert (
            journey.total_travel_minutes + journey.total_transfer_minutes
            == journey.total_duration_minutes
        )

    def test_same_train_split_is_never_classified_as_connecting(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for journey in journeys_of_type(result, JourneyType.SAME_TRAIN_SPLIT):
            assert journey.train_changes == 0
        for journey in journeys_of_type(result, JourneyType.CONNECTING):
            assert journey.train_changes >= 1

    def test_three_reservation_split_is_generated_for_a_four_stop_train(
        self, timetable, availability_book, config
    ) -> None:
        """YT1003 BLR -> HYD -> NGP -> DEL can be split at both middle stops."""
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journeys = [
            journey
            for journey in result.options
            if journey.journey_type is JourneyType.SAME_TRAIN_SPLIT
            and journey.train_numbers == ("YT1003", "YT1003", "YT1003")
            and journey.reservation_count == 3
        ]
        assert journeys
        journey = journeys[0]
        assert journey.station_path == ("BLR", "HYD", "NGP", "DEL")
        assert journey.train_changes == 0
        assert journey.total_fare == Fare.from_paise(90000 + 40000 + 100000)

    def test_split_risk_is_safe_because_the_train_does_not_change(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        from engine.enums import ConnectionKind, ConnectionRisk

        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = find_journey(
            result, train_numbers=("YT1001", "YT1001"), journey_type=JourneyType.SAME_TRAIN_SPLIT
        )
        assert journey is not None
        assert journey.connections[0].kind is ConnectionKind.SAME_TRAIN
        assert journey.connections[0].risk is ConnectionRisk.SAFE
        assert journey.connections[0].required_minutes == 0


class TestSplitPlanValidation:
    def test_plan_rejects_a_same_train_split_across_two_trains(self, timetable) -> None:
        yt1001 = timetable.train("YT1001")
        yt1005 = timetable.train("YT1005")
        from engine.legs import RailLeg

        first = RailLeg(train=yt1001, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE)
        second = RailLeg(train=yt1005, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE)
        with pytest.raises(DomainValidationError):
            JourneyPlan(JourneyType.SAME_TRAIN_SPLIT, (first, second))

    def test_plan_rejects_non_contiguous_segments(self, timetable) -> None:
        from engine.legs import RailLeg

        yt1003 = timetable.train("YT1003")
        first = RailLeg(train=yt1003, origin_index=0, destination_index=1, anchor_date=TRAVEL_DATE)
        second = RailLeg(train=yt1003, origin_index=2, destination_index=3, anchor_date=TRAVEL_DATE)
        with pytest.raises(DomainValidationError):
            JourneyPlan(JourneyType.CONNECTING, (first, second))
