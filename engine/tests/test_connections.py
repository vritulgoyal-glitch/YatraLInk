"""Connection validation and deterministic transfer-risk classification."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from engine.config import SearchConfiguration
from engine.connections import (
    ConnectionAssessment,
    assess_connection,
    classify_connection_kind,
    connection_risk_for_buffer,
    describe_connection_rule,
    validate_connection,
)
from engine.enums import ConnectionKind, ConnectionRisk, RejectionReason
from engine.errors import InvalidJourneyError
from engine.network import TransferAllowance
from engine.tests.builders import TRAVEL_DATE, at, make_segment, make_train

DAY = TRAVEL_DATE


@pytest.fixture()
def config() -> SearchConfiguration:
    """Default rules: 30 minute minimum, TIGHT up to +30 minutes of buffer."""
    return SearchConfiguration()


def _arrives_at(hour: int, minute: int = 0) -> datetime:
    return at(DAY, hour, minute)


def _departs_at(hour: int, minute: int = 0) -> datetime:
    return at(DAY, hour, minute)


class TestRiskClassification:
    def test_negative_buffer_is_invalid(self) -> None:
        assert (
            connection_risk_for_buffer(-1, tight_connection_max_buffer_minutes=30)
            is ConnectionRisk.INVALID
        )

    def test_zero_buffer_is_tight(self) -> None:
        assert (
            connection_risk_for_buffer(0, tight_connection_max_buffer_minutes=30)
            is ConnectionRisk.TIGHT
        )

    def test_buffer_at_the_tight_ceiling_is_tight(self) -> None:
        assert (
            connection_risk_for_buffer(30, tight_connection_max_buffer_minutes=30)
            is ConnectionRisk.TIGHT
        )

    def test_buffer_above_the_ceiling_is_safe(self) -> None:
        assert (
            connection_risk_for_buffer(31, tight_connection_max_buffer_minutes=30)
            is ConnectionRisk.SAFE
        )

    def test_threshold_is_configurable(self) -> None:
        assert (
            connection_risk_for_buffer(45, tight_connection_max_buffer_minutes=60)
            is ConnectionRisk.TIGHT
        )
        assert (
            connection_risk_for_buffer(61, tight_connection_max_buffer_minutes=60)
            is ConnectionRisk.SAFE
        )

    def test_assessment_rejects_an_inconsistent_verdict(self) -> None:
        with pytest.raises(ValueError):
            ConnectionAssessment(
                kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
                transfer_minutes=60,
                required_minutes=30,
                buffer_minutes=30,
                valid=True,
                risk=ConnectionRisk.INVALID,
            )


class TestSameStationConnections:
    """The worked examples from the specification."""

    def test_20_minutes_against_a_30_minute_minimum_is_invalid(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(10, 0),
            departure=_departs_at(10, 20),
            config=config,
        )
        assert assessment.valid is False
        assert assessment.risk is ConnectionRisk.INVALID
        assert assessment.transfer_minutes == 20
        assert assessment.required_minutes == 30
        assert assessment.buffer_minutes == -10

    def test_29_minutes_is_invalid(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(10, 0),
            departure=_departs_at(10, 29),
            config=config,
        )
        assert assessment.valid is False

    def test_30_minutes_is_valid_and_tight(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(10, 0),
            departure=_departs_at(10, 30),
            config=config,
        )
        assert assessment.valid is True
        assert assessment.risk is ConnectionRisk.TIGHT
        assert assessment.buffer_minutes == 0

    def test_31_minutes_is_valid_and_tight(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(10, 0),
            departure=_departs_at(10, 31),
            config=config,
        )
        assert assessment.risk is ConnectionRisk.TIGHT

    def test_61_minutes_is_safe(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(10, 0),
            departure=_departs_at(11, 1),
            config=config,
        )
        assert assessment.valid is True
        assert assessment.risk is ConnectionRisk.SAFE
        assert assessment.buffer_minutes == 31

    def test_zero_minute_connection_is_invalid(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(10, 0),
            departure=_departs_at(10, 0),
            config=config,
        )
        assert assessment.valid is False

    def test_negative_transfer_is_invalid(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(11, 0),
            departure=_departs_at(10, 0),
            config=config,
        )
        assert assessment.valid is False
        assert assessment.transfer_minutes == -60

    def test_custom_minimum_is_respected(self) -> None:
        strict = SearchConfiguration(minimum_connection_minutes=90)
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(10, 0),
            departure=_departs_at(11, 30),
            config=strict,
        )
        assert assessment.required_minutes == 90
        assert assessment.transfer_minutes == 90
        assert assessment.valid is True
        assert assessment.risk is ConnectionRisk.TIGHT  # buffer 0 -> tight

    def test_custom_minimum_still_rejects_a_shorter_transfer(self) -> None:
        strict = SearchConfiguration(minimum_connection_minutes=90)
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(10, 0),
            departure=_departs_at(11, 0),
            config=strict,
        )
        assert assessment.valid is False
        assert assessment.risk is ConnectionRisk.INVALID


class TestSameTrainConnections:
    def test_same_train_requires_no_transfer_time(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.SAME_TRAIN,
            arrival=_arrives_at(10, 0),
            departure=_arrives_at(10, 0),
            config=config,
        )
        assert assessment.valid is True
        assert assessment.required_minutes == 0
        assert assessment.risk is ConnectionRisk.SAFE

    def test_same_train_with_a_halt_is_safe(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.SAME_TRAIN,
            arrival=_arrives_at(5, 40),
            departure=_departs_at(5, 45),
            config=config,
        )
        assert assessment.risk is ConnectionRisk.SAFE
        assert assessment.transfer_minutes == 5

    def test_same_train_next_day_is_safe(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.SAME_TRAIN,
            arrival=_arrives_at(23, 0),
            departure=at(DAY + timedelta(days=1), 6, 0),
            config=config,
        )
        assert assessment.valid is True
        assert assessment.risk is ConnectionRisk.SAFE

    def test_same_train_backwards_is_invalid(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.SAME_TRAIN,
            arrival=_arrives_at(10, 0),
            departure=_arrives_at(9, 0),
            config=config,
        )
        assert assessment.valid is False


class TestOvernightConnections:
    def test_overnight_connection_uses_real_datetimes(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(23, 0),
            departure=at(DAY + timedelta(days=1), 0, 30),
            config=config,
        )
        assert assessment.transfer_minutes == 90
        assert assessment.valid is True
        assert assessment.risk is ConnectionRisk.SAFE

    def test_overnight_connection_below_minimum_is_invalid(self, config) -> None:
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(23, 50),
            departure=at(DAY + timedelta(days=1), 0, 10),
            config=config,
        )
        assert assessment.transfer_minutes == 20
        assert assessment.valid is False

    def test_clock_only_comparison_would_have_been_wrong(self, config) -> None:
        """10:00 next day is 24h after 10:00 today, not zero minutes."""
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
            arrival=_arrives_at(10, 0),
            departure=at(DAY + timedelta(days=1), 10, 0),
            config=config,
        )
        assert assessment.transfer_minutes == 1440
        assert assessment.risk is ConnectionRisk.SAFE


class TestCrossStationTransfers:
    def test_station_change_is_invalid_when_disabled(self, config) -> None:
        assert config.allow_cross_station_transfers is False
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_STATION_TRANSFER,
            arrival=_arrives_at(20, 0),
            departure=_departs_at(23, 0),
            config=config,
            allowance=TransferAllowance(
                from_station_code="NDLS", to_station_code="DEL", minimum_minutes=45
            ),
        )
        assert assessment.valid is False
        assert assessment.risk is ConnectionRisk.INVALID

    def test_station_change_without_an_allowance_is_invalid(self) -> None:
        enabled = SearchConfiguration(allow_cross_station_transfers=True)
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_STATION_TRANSFER,
            arrival=_arrives_at(20, 0),
            departure=_departs_at(23, 0),
            config=enabled,
            allowance=None,
        )
        assert assessment.valid is False
        assert "allowance" in assessment.note

    def test_station_change_with_an_allowance_is_valid_but_never_safe(self) -> None:
        enabled = SearchConfiguration(allow_cross_station_transfers=True)
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_STATION_TRANSFER,
            arrival=_arrives_at(20, 0),
            departure=_departs_at(23, 0),
            config=enabled,
            allowance=TransferAllowance(
                from_station_code="NDLS", to_station_code="DEL", minimum_minutes=45
            ),
        )
        assert assessment.valid is True
        assert assessment.risk is ConnectionRisk.TIGHT
        assert assessment.risk is not ConnectionRisk.SAFE
        assert assessment.required_minutes == 45

    def test_station_change_below_the_allowance_is_invalid(self) -> None:
        enabled = SearchConfiguration(allow_cross_station_transfers=True)
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_STATION_TRANSFER,
            arrival=_arrives_at(20, 0),
            departure=_departs_at(20, 30),
            config=enabled,
            allowance=TransferAllowance(
                from_station_code="NDLS", to_station_code="DEL", minimum_minutes=45
            ),
        )
        assert assessment.valid is False

    def test_allowance_takes_the_stricter_of_the_two_rules(self) -> None:
        enabled = SearchConfiguration(
            allow_cross_station_transfers=True, minimum_connection_minutes=60
        )
        assessment = assess_connection(
            kind=ConnectionKind.CROSS_STATION_TRANSFER,
            arrival=_arrives_at(20, 0),
            departure=_departs_at(20, 50),
            config=enabled,
            allowance=TransferAllowance(
                from_station_code="NDLS", to_station_code="DEL", minimum_minutes=45
            ),
        )
        assert assessment.required_minutes == 60
        assert assessment.valid is False

    def test_allowances_are_directional(self) -> None:
        from engine.network import lookup_transfer_allowance

        allowances = (
            TransferAllowance(from_station_code="NDLS", to_station_code="DEL", minimum_minutes=45),
        )
        assert lookup_transfer_allowance("NDLS", "DEL", allowances) is not None
        assert lookup_transfer_allowance("DEL", "NDLS", allowances) is None

    def test_ndls_and_del_are_not_silently_treated_as_the_same_station(self, timetable) -> None:
        """Two distinct station codes must never be merged by the engine."""
        assert timetable.station("NDLS").code != timetable.station("DEL").code
        assert timetable.station("NDLS").name != timetable.station("DEL").name


class TestConnectionKindClassification:
    def test_same_train_kind(self) -> None:
        train = make_train("YT9001", [("BLR", None, "06:00", 0), ("HYD", "10:00", None, 0)])
        first = make_segment(train, "BLR", "HYD")
        second = make_segment(train, "BLR", "HYD", reservation_index=1)
        assert classify_connection_kind(first, second) is ConnectionKind.SAME_TRAIN

    def test_same_station_different_train_kind(self) -> None:
        first_train = make_train("YT9001", [("BLR", None, "06:00", 0), ("HYD", "10:00", None, 0)])
        second_train = make_train("YT9002", [("HYD", None, "11:00", 0), ("DEL", "20:00", None, 0)])
        assert (
            classify_connection_kind(
                make_segment(first_train, "BLR", "HYD"), make_segment(second_train, "HYD", "DEL")
            )
            is ConnectionKind.CROSS_TRAIN_SAME_STATION
        )

    def test_station_change_kind(self) -> None:
        first_train = make_train("YT9001", [("BLR", None, "06:00", 0), ("NDLS", "20:00", None, 0)])
        # A station-change transfer is often an overnight connection, so the
        # second train's first stop is anchored to the following day.
        second_train = make_train("YT9002", [("DEL", None, "23:00", 0), ("BPL", "06:00", None, 1)])
        assert (
            classify_connection_kind(
                make_segment(first_train, "BLR", "NDLS"), make_segment(second_train, "DEL", "BPL")
            )
            is ConnectionKind.CROSS_STATION_TRANSFER
        )


class TestValidateConnection:
    def test_valid_connection_returns_info(self, config) -> None:
        first_train = make_train("YT9001", [("BLR", None, "06:00", 0), ("HYD", "10:00", None, 0)])
        second_train = make_train("YT9002", [("HYD", None, "11:00", 0), ("DEL", "20:00", None, 0)])
        info = validate_connection(
            index=0,
            previous=make_segment(first_train, "BLR", "HYD"),
            following=make_segment(second_train, "HYD", "DEL"),
            config=config,
        )
        assert info.valid is True
        assert info.transfer_minutes == 60
        assert info.required_minutes == 30
        assert info.buffer_minutes == 30
        assert info.risk is ConnectionRisk.TIGHT  # buffer 30 is exactly the ceiling
        assert info.from_station_code == info.to_station_code == "HYD"

    def test_a_generous_buffer_is_safe(self, config) -> None:
        first_train = make_train("YT9001", [("BLR", None, "06:00", 0), ("HYD", "10:00", None, 0)])
        second_train = make_train("YT9002", [("HYD", None, "12:00", 0), ("DEL", "20:00", None, 0)])
        info = validate_connection(
            index=0,
            previous=make_segment(first_train, "BLR", "HYD"),
            following=make_segment(second_train, "HYD", "DEL"),
            config=config,
        )
        assert info.transfer_minutes == 120
        assert info.buffer_minutes == 90
        assert info.risk is ConnectionRisk.SAFE

    def test_invalid_connection_raises_with_a_reason(self, config) -> None:
        first_train = make_train("YT9001", [("BLR", None, "06:00", 0), ("HYD", "10:00", None, 0)])
        second_train = make_train("YT9002", [("HYD", None, "10:20", 0), ("DEL", "20:00", None, 0)])
        with pytest.raises(InvalidJourneyError) as error:
            validate_connection(
                index=0,
                previous=make_segment(first_train, "BLR", "HYD"),
                following=make_segment(second_train, "HYD", "DEL"),
                config=config,
            )
        assert error.value.reason is RejectionReason.INVALID_CONNECTION

    def test_rule_description_mentions_the_minimum(self, config) -> None:
        assert "30" in describe_connection_rule(config)
