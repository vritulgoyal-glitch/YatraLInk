"""Domain model invariants: stations, stops, trains, segments and journeys."""

from __future__ import annotations

from datetime import time

import pytest

from engine.enums import AvailabilityState, ConnectionKind, ConnectionRisk, JourneyType
from engine.errors import DomainValidationError, InvalidJourneyError
from engine.models import (
    ConnectionInfo,
    JourneyOption,
    JourneySegment,
    Station,
    Train,
    TrainStop,
    normalize_station_code,
    normalize_train_number,
)
from engine.money import Fare
from engine.tests.builders import make_segment, make_train


class TestStationCodeNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("BLR", "BLR"), ("blr", "BLR"), ("  Blr  ", "BLR"), ("ndls", "NDLS"), ("bpl1", "BPL1")],
    )
    def test_normalises_case_and_whitespace(self, raw: str, expected: str) -> None:
        assert normalize_station_code(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "1BLR", "B", "B@LR", "B LR", "BLR-1"])
    def test_rejects_invalid_codes(self, raw: str) -> None:
        with pytest.raises(DomainValidationError):
            normalize_station_code(raw)

    def test_rejects_non_string(self) -> None:
        with pytest.raises(DomainValidationError):
            normalize_station_code(None)  # type: ignore[arg-type]

    def test_station_name_is_not_assumed_unique(self) -> None:
        """Two different stations may legitimately share a display name."""
        first = Station(code="AAA", name="Sample Junction")
        second = Station(code="BBB", name="Sample Junction")
        assert first.name == second.name
        assert first.code != second.code

    def test_station_requires_a_name(self) -> None:
        with pytest.raises(DomainValidationError):
            Station(code="BLR", name="   ")

    def test_train_number_normalisation(self) -> None:
        assert normalize_train_number(" yt1001 ") == "YT1001"

    @pytest.mark.parametrize("raw", ["", " ", "-x", "A"])
    def test_invalid_train_numbers(self, raw: str) -> None:
        with pytest.raises(DomainValidationError):
            normalize_train_number(raw)


class TestTrainStop:
    def test_origin_stop_needs_only_a_departure(self) -> None:
        stop = TrainStop(station_code="BLR", sequence=1, departure=time(23, 30))
        assert stop.arrival is None
        assert stop.departure_day_offset == 0

    def test_terminus_stop_needs_only_an_arrival(self) -> None:
        stop = TrainStop(station_code="DEL", sequence=3, arrival=time(22, 0), day_offset=1)
        assert stop.departure is None

    def test_stop_without_any_time_is_rejected(self) -> None:
        with pytest.raises(DomainValidationError):
            TrainStop(station_code="BLR", sequence=1)

    def test_departure_before_arrival_is_rejected(self) -> None:
        with pytest.raises(DomainValidationError):
            TrainStop(
                station_code="HYD",
                sequence=2,
                arrival=time(10, 0),
                departure=time(9, 0),
            )

    def test_halt_crossing_midnight_is_allowed(self) -> None:
        stop = TrainStop(
            station_code="HYD",
            sequence=2,
            arrival=time(23, 50),
            departure=time(0, 10),
            day_offset=0,
            departure_day_offset=1,
        )
        assert stop.halt_minutes == 20

    def test_departure_day_offset_beyond_one_day_is_rejected(self) -> None:
        with pytest.raises(DomainValidationError):
            TrainStop(
                station_code="HYD",
                sequence=2,
                arrival=time(23, 50),
                departure=time(0, 10),
                day_offset=0,
                departure_day_offset=2,
            )

    def test_sub_minute_precision_is_rejected(self) -> None:
        with pytest.raises(DomainValidationError):
            TrainStop(station_code="BLR", sequence=1, departure=time(23, 30, 30))

    def test_sequence_must_be_positive(self) -> None:
        with pytest.raises(DomainValidationError):
            TrainStop(station_code="BLR", sequence=0, departure=time(1, 0))


class TestTrain:
    def test_route_order_is_preserved(self) -> None:
        train = make_train("YT9001", [("BLR", None, "23:30", 0), ("HYD", "05:40", None, 1)])
        assert train.station_codes == ("BLR", "HYD")
        assert train.origin_code == "BLR"
        assert train.terminus_code == "HYD"

    def test_serves_in_order_is_directional(self) -> None:
        train = make_train(
            "YT9001",
            [("BLR", None, "23:30", 0), ("HYD", "05:40", "05:45", 1), ("DEL", "22:00", None, 1)],
        )
        assert train.serves_in_order("BLR", "DEL")
        assert not train.serves_in_order("DEL", "BLR")

    def test_reverse_direction_is_reported_as_not_served(self) -> None:
        train = make_train(
            "YT9001",
            [("BLR", None, "23:30", 0), ("HYD", "05:40", "05:45", 1), ("DEL", "22:00", None, 1)],
        )
        assert train.stops_between("DEL", "BLR") == ()
        assert train.stops_between("BLR", "BLR") == ()

    def test_non_chronological_route_is_rejected(self) -> None:
        with pytest.raises(DomainValidationError):
            make_train("YT9002", [("BLR", None, "10:00", 0), ("HYD", "08:00", None, 0)])

    def test_repeated_station_in_route_is_rejected(self) -> None:
        with pytest.raises(DomainValidationError):
            make_train(
                "YT9003",
                [
                    ("BLR", None, "06:00", 0),
                    ("HYD", "10:00", "10:10", 0),
                    ("BLR", "14:00", None, 0),
                ],
            )

    def test_single_stop_route_is_rejected(self) -> None:
        with pytest.raises(DomainValidationError):
            make_train("YT9004", [("BLR", None, "06:00", 0)])

    def test_origin_requires_departure_and_terminus_requires_arrival(self) -> None:
        with pytest.raises(DomainValidationError):
            make_train("YT9005", [("BLR", "06:00", None, 0), ("HYD", "08:00", None, 0)])

    def test_intermediate_stop_requires_both_times(self) -> None:
        with pytest.raises(DomainValidationError):
            make_train(
                "YT9006",
                [
                    ("BLR", None, "06:00", 0),
                    ("HYD", "10:00", None, 0),
                    ("DEL", "18:00", None, 0),
                ],
            )


class TestJourneySegment:
    def test_valid_segment(self) -> None:
        train = make_train("YT9001", [("BLR", None, "23:30", 0), ("HYD", "05:40", None, 1)])
        segment = make_segment(train, "BLR", "HYD", fare_paise=98000)
        assert segment.duration_minutes == 370
        assert segment.fare == Fare.from_paise(98000)
        assert segment.origin_station_code == "BLR"

    def test_identical_origin_and_destination_is_rejected(self) -> None:
        with pytest.raises(InvalidJourneyError):
            JourneySegment(
                train_number="YT9001",
                train_name="x",
                origin_station_code="BLR",
                destination_station_code="BLR",
                departure=TrainStop(
                    station_code="BLR", sequence=1, departure=time(6, 0)
                ).departure_datetime,  # type: ignore[arg-type]
                arrival=TrainStop(
                    station_code="BLR", sequence=1, departure=time(7, 0)
                ).departure_datetime,  # type: ignore[arg-type]
                duration_minutes=60,
                availability=AvailabilityState.AVAILABLE,
            )

    def test_duration_must_match_the_times(self) -> None:
        train = make_train("YT9001", [("BLR", None, "23:30", 0), ("HYD", "05:40", None, 1)])
        leg = make_segment(train, "BLR", "HYD")
        with pytest.raises(InvalidJourneyError):
            JourneySegment(
                train_number=leg.train_number,
                train_name=leg.train_name,
                origin_station_code=leg.origin_station_code,
                destination_station_code=leg.destination_station_code,
                departure=leg.departure,
                arrival=leg.arrival,
                duration_minutes=1,
                availability=AvailabilityState.AVAILABLE,
            )

    def test_zero_duration_is_rejected(self) -> None:
        with pytest.raises(InvalidJourneyError):
            JourneySegment(
                train_number="YT9001",
                train_name="x",
                origin_station_code="BLR",
                destination_station_code="HYD",
                departure=TrainStop(
                    station_code="BLR", sequence=1, departure=time(6, 0)
                ).departure_datetime,  # type: ignore[arg-type]
                arrival=TrainStop(
                    station_code="BLR", sequence=1, departure=time(6, 0)
                ).departure_datetime,  # type: ignore[arg-type]
                duration_minutes=1,
                availability=AvailabilityState.AVAILABLE,
            )

    def test_unknown_availability_string_is_rejected(self) -> None:
        train = make_train("YT9001", [("BLR", None, "06:00", 0), ("HYD", "08:00", None, 0)])
        leg = make_segment(train, "BLR", "HYD")
        with pytest.raises(DomainValidationError):
            JourneySegment(
                train_number=leg.train_number,
                train_name=leg.train_name,
                origin_station_code=leg.origin_station_code,
                destination_station_code=leg.destination_station_code,
                departure=leg.departure,
                arrival=leg.arrival,
                duration_minutes=leg.duration_minutes,
                availability="CONFIRMED",  # type: ignore[arg-type]
            )


class TestJourneyOption:
    def _three_stop_train(self) -> Train:
        return make_train(
            "YT9001",
            [("BLR", None, "23:30", 0), ("HYD", "05:40", "05:45", 1), ("DEL", "22:00", None, 1)],
        )

    def test_direct_journey_requires_exactly_one_segment(self) -> None:
        train = self._three_stop_train()
        with pytest.raises(InvalidJourneyError):
            JourneyOption(
                journey_id="x",
                journey_type=JourneyType.DIRECT,
                origin_station_code="BLR",
                destination_station_code="DEL",
                segments=(
                    make_segment(train, "BLR", "HYD"),
                    make_segment(train, "HYD", "DEL", reservation_index=1),
                ),
            )

    def test_same_train_split_requires_one_train_and_two_segments(self) -> None:
        train = self._three_stop_train()
        option = JourneyOption(
            journey_id="split",
            journey_type=JourneyType.SAME_TRAIN_SPLIT,
            origin_station_code="BLR",
            destination_station_code="DEL",
            segments=(
                make_segment(train, "BLR", "HYD"),
                make_segment(train, "HYD", "DEL", reservation_index=1),
            ),
        )
        assert option.train_changes == 0
        assert option.reservation_count == 2
        assert option.requires_separate_reservations

    def test_same_train_split_across_two_trains_is_rejected(self) -> None:
        first = make_train("YT9001", [("BLR", None, "23:30", 0), ("HYD", "05:40", "05:45", 1)])
        second = make_train("YT9002", [("HYD", None, "05:45", 0), ("DEL", "22:00", None, 0)])
        with pytest.raises(InvalidJourneyError):
            JourneyOption(
                journey_id="bad",
                journey_type=JourneyType.SAME_TRAIN_SPLIT,
                origin_station_code="BLR",
                destination_station_code="DEL",
                segments=(
                    make_segment(first, "BLR", "HYD"),
                    make_segment(second, "HYD", "DEL", reservation_index=1),
                ),
            )

    def test_connecting_journey_counts_train_changes(self) -> None:
        first = make_train("YT9001", [("BLR", None, "23:30", 0), ("HYD", "05:40", "05:45", 1)])
        second = make_train("YT9002", [("HYD", None, "06:45", 0), ("DEL", "20:00", None, 0)])
        option = JourneyOption(
            journey_id="connect",
            journey_type=JourneyType.CONNECTING,
            origin_station_code="BLR",
            destination_station_code="DEL",
            segments=(
                make_segment(first, "BLR", "HYD"),
                make_segment(second, "HYD", "DEL", reservation_index=1),
            ),
        )
        assert option.train_changes == 1
        assert option.total_fare == Fare.from_paise(200000)

    def test_total_duration_is_elapsed_not_summed_segment_durations(self) -> None:
        """08:00-12:00 then 14:00-18:00 is 10 hours, not 8."""
        first = make_train("YT9101", [("AAA", None, "08:00", 0), ("BBB", "12:00", None, 0)])
        second = make_train("YT9102", [("BBB", None, "14:00", 0), ("CCC", "18:00", None, 0)])
        option = JourneyOption(
            journey_id="elapsed",
            journey_type=JourneyType.CONNECTING,
            origin_station_code="AAA",
            destination_station_code="CCC",
            segments=(
                make_segment(first, "AAA", "BBB"),
                make_segment(second, "BBB", "CCC", reservation_index=1),
            ),
        )
        assert option.total_duration_minutes == 600
        assert option.total_travel_minutes == 480
        assert option.total_transfer_minutes == 120

    def test_more_than_three_segments_is_rejected(self) -> None:
        train = make_train(
            "YT9001",
            [
                ("AAA", None, "06:00", 0),
                ("BBB", "08:00", "08:10", 0),
                ("CCC", "10:00", "10:10", 0),
                ("DDD", "12:00", "12:10", 0),
                ("EEE", "14:00", None, 0),
            ],
        )
        with pytest.raises(InvalidJourneyError):
            JourneyOption(
                journey_id="toolong",
                journey_type=JourneyType.SAME_TRAIN_SPLIT,
                origin_station_code="AAA",
                destination_station_code="EEE",
                segments=(
                    make_segment(train, "AAA", "BBB"),
                    make_segment(train, "BBB", "CCC", reservation_index=1),
                    make_segment(train, "CCC", "DDD", reservation_index=2),
                    make_segment(train, "DDD", "EEE", reservation_index=3),
                ),
            )

    def test_empty_segment_list_is_rejected(self) -> None:
        with pytest.raises(InvalidJourneyError):
            JourneyOption(
                journey_id="empty",
                journey_type=JourneyType.DIRECT,
                origin_station_code="BLR",
                destination_station_code="DEL",
                segments=(),
            )

    def test_dedup_key_includes_journey_type(self) -> None:
        train = self._three_stop_train()
        segments = (
            make_segment(train, "BLR", "HYD"),
            make_segment(train, "HYD", "DEL", reservation_index=1),
        )
        split = JourneyOption(
            journey_id="a",
            journey_type=JourneyType.SAME_TRAIN_SPLIT,
            origin_station_code="BLR",
            destination_station_code="DEL",
            segments=segments,
        )
        connecting = JourneyOption(
            journey_id="b",
            journey_type=JourneyType.CONNECTING,
            origin_station_code="BLR",
            destination_station_code="DEL",
            segments=segments,
        )
        assert split.dedup_key != connecting.dedup_key

    def test_unknown_fare_is_none_not_zero(self) -> None:
        train = make_train("YT9001", [("BLR", None, "06:00", 0), ("HYD", "08:00", None, 0)])
        option = JourneyOption(
            journey_id="nofare",
            journey_type=JourneyType.DIRECT,
            origin_station_code="BLR",
            destination_station_code="HYD",
            segments=(make_segment(train, "BLR", "HYD", fare_paise=None),),
        )
        assert option.total_fare is None

    def test_any_unknown_fare_makes_the_total_unknown(self) -> None:
        first = make_train("YT9001", [("BLR", None, "23:30", 0), ("HYD", "05:40", "05:45", 1)])
        second = make_train("YT9002", [("HYD", None, "06:45", 0), ("DEL", "20:00", None, 0)])
        option = JourneyOption(
            journey_id="partial",
            journey_type=JourneyType.CONNECTING,
            origin_station_code="BLR",
            destination_station_code="DEL",
            segments=(
                make_segment(first, "BLR", "HYD", fare_paise=100000),
                make_segment(second, "HYD", "DEL", fare_paise=None, reservation_index=1),
            ),
        )
        assert option.total_fare is None

    def test_station_path_includes_transfer_points(self) -> None:
        train = self._three_stop_train()
        option = JourneyOption(
            journey_id="path",
            journey_type=JourneyType.SAME_TRAIN_SPLIT,
            origin_station_code="BLR",
            destination_station_code="DEL",
            segments=(
                make_segment(train, "BLR", "HYD"),
                make_segment(train, "HYD", "DEL", reservation_index=1),
            ),
        )
        assert option.station_path == ("BLR", "HYD", "DEL")


class TestConnectionInfo:
    def test_station_change_flag(self) -> None:
        info = ConnectionInfo(
            index=0,
            kind=ConnectionKind.CROSS_STATION_TRANSFER,
            arrival=TrainStop(
                station_code="NDLS", sequence=1, departure=time(20, 0)
            ).departure_datetime,  # type: ignore[arg-type]
            departure=TrainStop(
                station_code="DEL", sequence=1, departure=time(22, 0)
            ).departure_datetime,  # type: ignore[arg-type]
            transfer_minutes=120,
            required_minutes=45,
            valid=True,
            risk=ConnectionRisk.TIGHT,
            from_station_code="NDLS",
            to_station_code="DEL",
            buffer_minutes=75,
        )
        assert info.station_change
        assert not info.is_same_station
