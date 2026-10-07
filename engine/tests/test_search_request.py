"""Search request validation and normalisation."""

from __future__ import annotations

from datetime import date, datetime, time

import pytest

from engine.config import SearchConfiguration
from engine.enums import TravelClass
from engine.errors import SearchRequestError
from engine.ranking import normalise
from engine.search import SearchRequest

TRAVEL_DATE = date(2026, 6, 15)


class TestValidRequests:
    def test_minimal_request_defaults(self) -> None:
        request = SearchRequest(origin="blr", destination="del", travel_date=TRAVEL_DATE)
        assert request.origin == "BLR"
        assert request.destination == "DEL"
        assert request.passengers == 1
        assert request.travel_class is TravelClass.AC_3_TIER
        assert request.quota == "GENERAL"

    def test_station_codes_are_normalised(self) -> None:
        request = SearchRequest(origin=" blr ", destination=" Del ", travel_date=TRAVEL_DATE)
        assert (request.origin, request.destination) == ("BLR", "DEL")

    def test_class_accepts_a_code_string(self) -> None:
        request = SearchRequest(
            origin="BLR", destination="DEL", travel_date=TRAVEL_DATE, travel_class="SL"
        )
        assert request.travel_class is TravelClass.SLEEPER

    def test_quota_is_upper_cased(self) -> None:
        request = SearchRequest(
            origin="BLR", destination="DEL", travel_date=TRAVEL_DATE, quota="tatkal"
        )
        assert request.quota == "TATKAL"

    def test_earliest_departure_resolves_onto_the_travel_date(self) -> None:
        request = SearchRequest(
            origin="BLR",
            destination="DEL",
            travel_date=TRAVEL_DATE,
            earliest_departure=time(8, 30),
        )
        assert request.earliest_departure_datetime == datetime(2026, 6, 15, 8, 30)

    def test_start_of_travel_date_is_midnight(self) -> None:
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        assert request.start_of_travel_date == datetime(2026, 6, 15, 0, 0)

    def test_max_train_changes_of_two_is_allowed(self) -> None:
        request = SearchRequest(
            origin="BLR", destination="DEL", travel_date=TRAVEL_DATE, max_train_changes=2
        )
        assert request.max_train_changes == 2

    def test_passenger_upper_bound(self) -> None:
        request = SearchRequest(
            origin="BLR", destination="DEL", travel_date=TRAVEL_DATE, passengers=6
        )
        assert request.passengers == 6

    def test_describe_is_readable(self) -> None:
        request = SearchRequest(
            origin="BLR", destination="DEL", travel_date=TRAVEL_DATE, passengers=2
        )
        assert "BLR->DEL" in request.describe()
        assert "2pax" in request.describe()


class TestInvalidRequests:
    def test_same_origin_and_destination_is_rejected(self) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(origin="BLR", destination="blr", travel_date=TRAVEL_DATE)

    @pytest.mark.parametrize("code", ["", "   ", "1BLR", "B"])
    def test_empty_or_invalid_station_codes_are_rejected(self, code: str) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(origin=code, destination="DEL", travel_date=TRAVEL_DATE)

    def test_missing_origin_is_rejected(self) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(origin=None, destination="DEL", travel_date=TRAVEL_DATE)  # type: ignore[arg-type]

    def test_missing_travel_date_is_rejected(self) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(origin="BLR", destination="DEL", travel_date=None)  # type: ignore[arg-type]

    def test_datetime_is_rejected_where_a_date_is_required(self) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(
                origin="BLR",
                destination="DEL",
                travel_date=datetime(2026, 6, 15, 6, 0),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("count", [0, -1, 7, 100])
    def test_invalid_passenger_counts_are_rejected(self, count: int) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(
                origin="BLR", destination="DEL", travel_date=TRAVEL_DATE, passengers=count
            )

    def test_boolean_passenger_count_is_rejected(self) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(
                origin="BLR",
                destination="DEL",
                travel_date=TRAVEL_DATE,
                passengers=True,  # type: ignore[arg-type]
            )

    def test_unsupported_travel_class_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            SearchRequest(
                origin="BLR",
                destination="DEL",
                travel_date=TRAVEL_DATE,
                travel_class="FIRST_CLASS_LUXURY",  # type: ignore[arg-type]
            )

    def test_empty_quota_is_rejected(self) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE, quota="  ")

    def test_latest_arrival_must_be_after_the_travel_date(self) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(
                origin="BLR",
                destination="DEL",
                travel_date=TRAVEL_DATE,
                latest_arrival=datetime(2026, 6, 14, 23, 0),
            )

    def test_latest_arrival_must_be_after_earliest_departure(self) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(
                origin="BLR",
                destination="DEL",
                travel_date=TRAVEL_DATE,
                earliest_departure=time(18, 0),
                latest_arrival=datetime(2026, 6, 15, 17, 0),
            )

    def test_more_than_two_train_changes_is_rejected(self) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(
                origin="BLR", destination="DEL", travel_date=TRAVEL_DATE, max_train_changes=3
            )

    def test_sub_minute_earliest_departure_is_rejected(self) -> None:
        with pytest.raises(SearchRequestError):
            SearchRequest(
                origin="BLR",
                destination="DEL",
                travel_date=TRAVEL_DATE,
                earliest_departure=time(8, 30, 15),
            )


class TestSearchConfiguration:
    def test_defaults_match_the_phase_one_specification(self) -> None:
        config = SearchConfiguration()
        assert config.minimum_connection_minutes == 30
        assert config.tight_connection_max_buffer_minutes == 30
        assert config.max_train_changes == 2
        assert config.max_segments == 3
        assert config.allow_cross_station_transfers is False
        assert config.include_not_available_journeys is False

    def test_more_than_two_changes_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            SearchConfiguration(max_train_changes=3)

    def test_more_than_three_segments_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            SearchConfiguration(max_segments=4)

    def test_all_zero_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            SearchConfiguration(
                ranking_weights=__import__(
                    "engine.config", fromlist=["RankingWeights"]
                ).RankingWeights(
                    availability=0,
                    train_changes=0,
                    fare=0,
                    duration=0,
                    connection_risk=0,
                    separate_reservations=0,
                    station_change=0,
                )
            )

    def test_required_buffers_summary(self) -> None:
        buffers = SearchConfiguration().required_buffers
        assert buffers["SAME_TRAIN"] == 0
        assert buffers["CROSS_TRAIN_SAME_STATION"] == 30

    def test_to_json_is_serialisable(self) -> None:
        import json

        payload = SearchConfiguration().to_json()
        assert json.loads(json.dumps(payload))["max_segments"] == 3


class TestNormalise:
    """The integer scoring normaliser used by ranking."""

    def test_best_value_maps_to_the_maximum(self) -> None:
        assert normalise(10, best=10, worst=100, higher_is_better=False) == 1000

    def test_worst_value_maps_to_zero(self) -> None:
        assert normalise(100, best=10, worst=100, higher_is_better=False) == 0

    def test_midpoint_is_exact_integer(self) -> None:
        assert normalise(55, best=10, worst=100, higher_is_better=False) == 500

    def test_equal_range_scores_everyone_the_same(self) -> None:
        assert normalise(42, best=42, worst=42, higher_is_better=False) == 1000

    def test_out_of_range_values_are_clamped(self) -> None:
        assert normalise(-5, best=0, worst=100, higher_is_better=False) == 1000
        assert normalise(500, best=0, worst=100, higher_is_better=False) == 0

    def test_no_float_is_returned(self) -> None:
        assert isinstance(normalise(37, best=10, worst=100, higher_is_better=False), int)
