"""The normalized railway data contract: models, invariants, and engine bridging."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

import pytest

from engine import time_utils as tu
from engine.availability import AvailabilityState
from engine.enums import TravelClass
from engine.errors import DomainValidationError, UnknownStationError
from engine.journey import build_segment
from engine.legs import RailLeg
from engine.money import Fare
from engine.network import Timetable
from engine.provider import ProviderAvailabilityBook
from engine.railway_data import (
    AvailabilitySnapshot,
    Quota,
    RailwayStationStop,
    Station,
    Train,
    TrainClass,
    TrainSchedule,
    coerce_quota,
    normalize_train_class,
    require_ordered_route,
    require_paise,
)
from engine.search import SearchRequest

TZ = "Asia/Kolkata"
SERVICE_DATE = date(2026, 6, 15)
TRAVEL_DATE = date(2026, 7, 1)
FETCHED_AT = datetime(2026, 6, 30, 6, 15, tzinfo=UTC)
UPDATED_AT = datetime(2026, 6, 30, 5, 45, tzinfo=UTC)


# ----------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------
def stop(
    code: str,
    sequence: int,
    *,
    arrival: str | None = None,
    departure: str | None = None,
    day_offset: int = 0,
    departure_day_offset: int | None = None,
) -> RailwayStationStop:
    """Build a contract stop from readable clock strings."""
    return RailwayStationStop(
        station_code=code,
        sequence=sequence,
        arrival=tu.parse_clock(arrival) if arrival else None,
        departure=tu.parse_clock(departure) if departure else None,
        day_offset=day_offset,
        departure_day_offset=departure_day_offset,
    )


def overnight_train() -> Train:
    """``BLR`` (day 0, 21:40) -> ``HYD`` (day 1, 04:30) -> ``DEL`` (day 1, 18:05)."""
    return Train(
        train_number="YT1001",
        train_name="Sample Overnight Express",
        stops=(
            stop("BLR", 1, departure="21:40"),
            stop("HYD", 2, arrival="04:30", departure="04:50", day_offset=1),
            stop("DEL", 3, arrival="18:05", day_offset=1),
        ),
    )


def snapshot(**overrides: object) -> AvailabilitySnapshot:
    """A valid ``AVAILABLE`` snapshot, with any field overridable."""
    fields: dict[str, object] = {
        "train_number": "YT1001",
        "origin_station_code": "blr",
        "destination_station_code": "DEL",
        "travel_date": TRAVEL_DATE,
        "travel_class": "3A",
        "provider": "unit-test-source",
        "fetched_at": FETCHED_AT,
        "state": AvailabilityState.AVAILABLE,
        "quota": "GENERAL",
        "available_count": 42,
        "fare_paise": 245000,
        "updated_at": UPDATED_AT,
    }
    fields.update(overrides)
    return AvailabilitySnapshot(**fields)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# stations
# ----------------------------------------------------------------------
def test_valid_station_normalises_and_keeps_optional_fields() -> None:
    station = Station(code=" blr ", name="  KSR Bengaluru  ", city="Bengaluru", timezone=TZ)
    assert station.code == "BLR"
    assert station.name == "KSR Bengaluru"
    assert station.city == "Bengaluru"
    assert station.timezone == TZ
    assert station.tzinfo is not None
    assert station.to_engine_station().code == "BLR"
    assert station.to_json()["timezone"] == TZ


def test_station_without_optional_fields_is_valid() -> None:
    station = Station(code="DEL", name="Delhi")
    assert station.city is None
    assert station.timezone is None
    assert station.tzinfo is None


@pytest.mark.parametrize("code", ["", "   ", "9ABC", "A", "TOOLONGSTATIONCODE", "BL R"])
def test_invalid_station_code_is_rejected(code: str) -> None:
    with pytest.raises(DomainValidationError):
        Station(code=code, name="Whatever")


@pytest.mark.parametrize(
    ("name", "city", "timezone_name"),
    [("", None, None), ("Ok", "", None), ("Ok", None, "Not/AZone"), ("Ok", None, "")],
)
def test_invalid_station_text_or_timezone_is_rejected(
    name: str, city: str | None, timezone_name: str | None
) -> None:
    with pytest.raises(DomainValidationError):
        Station(code="BLR", name=name, city=city, timezone=timezone_name)


# ----------------------------------------------------------------------
# trains and schedules
# ----------------------------------------------------------------------
def test_valid_train_preserves_ordered_stops_and_day_offsets() -> None:
    train = overnight_train()
    assert train.train_number == "YT1001"
    assert train.station_codes == ("BLR", "HYD", "DEL")
    assert [s.sequence for s in train.stops] == [1, 2, 3]
    assert train.origin_code == "BLR"
    assert train.terminus_code == "DEL"
    assert train.stop_for("HYD").day_offset == 1
    assert train.serves_in_order("BLR", "DEL") is True
    assert train.serves_in_order("DEL", "BLR") is False
    assert train.stops_between("HYD", "DEL")[0].station_code == "HYD"
    with pytest.raises(UnknownStationError):
        train.stop_for("GOA")


def test_valid_train_schedule_anchors_stops_to_the_service_date() -> None:
    train = overnight_train()
    schedule = TrainSchedule(train=train, service_date=SERVICE_DATE)
    assert schedule.train_number == "YT1001"
    assert schedule.resolved_stops == train.stops
    assert schedule.origin_code == "BLR"
    assert schedule.terminus_code == "DEL"
    assert schedule.service_date_for(schedule.stop_for("HYD")) == SERVICE_DATE + timedelta(days=1)

    departure = schedule.departure_at("BLR", TZ)
    arrival = schedule.arrival_at("DEL", TZ)
    assert departure is not None and arrival is not None
    assert departure.isoformat() == "2026-06-15T21:40:00+05:30"
    assert arrival.isoformat() == "2026-06-16T18:05:00+05:30"
    assert schedule.arrival_at("BLR", TZ) is None
    assert schedule.departure_at("DEL", TZ) is None
    assert schedule.to_engine_train().station_codes == ("BLR", "HYD", "DEL")


def test_schedule_rejects_stops_that_contradict_the_train_route() -> None:
    train = overnight_train()
    with pytest.raises(DomainValidationError, match="do not match train"):
        TrainSchedule(
            train=train,
            service_date=SERVICE_DATE,
            stops=(stop("BLR", 1, departure="21:40"), stop("GOA", 2, arrival="23:00")),
        )


def test_schedule_rejects_a_non_date_service_date() -> None:
    with pytest.raises(DomainValidationError):
        TrainSchedule(train=overnight_train(), service_date=datetime(2026, 6, 15, 9, 0))  # type: ignore[arg-type]


def test_schedule_requires_a_train() -> None:
    with pytest.raises(DomainValidationError, match="must be a Train"):
        TrainSchedule(train=object(), service_date=SERVICE_DATE)  # type: ignore[arg-type]


def test_invalid_stop_sequence_is_rejected() -> None:
    with pytest.raises(DomainValidationError, match="sequence must be >= 1"):
        stop("BLR", 0, departure="21:40")
    with pytest.raises(DomainValidationError, match="strictly increasing"):
        Train(
            train_number="YT2001",
            train_name="Out Of Order",
            stops=(
                stop("BLR", 2, departure="21:40"),
                stop("DEL", 1, arrival="06:00", day_offset=1),
            ),
        )


def test_duplicate_station_sequence_is_rejected() -> None:
    with pytest.raises(DomainValidationError, match="more than once"):
        Train(
            train_number="YT2002",
            train_name="Loops Back",
            stops=(
                stop("BLR", 1, departure="21:40"),
                stop("HYD", 2, arrival="04:30", departure="04:50", day_offset=1),
                stop("BLR", 3, arrival="18:05", day_offset=1),
            ),
        )


def test_train_needs_at_least_two_stops_and_a_name() -> None:
    with pytest.raises(DomainValidationError, match="at least 2 stops"):
        Train(
            train_number="YT2003",
            train_name="Single",
            stops=(stop("BLR", 1, departure="21:40"),),
        )
    with pytest.raises(DomainValidationError, match="non-empty string"):
        Train(train_number="YT2004", train_name="", stops=overnight_train().stops)
    with pytest.raises(DomainValidationError, match="stops must be RailwayStationStop"):
        Train(train_number="YT2005", train_name="Bad stop", stops=("BLR", "DEL"))  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# day offsets and stop-time validation
# ----------------------------------------------------------------------
def test_day_offset_keeps_an_overnight_arrival_unambiguous() -> None:
    arrival = stop("HYD", 2, arrival="04:30", departure="04:50", day_offset=1)
    assert arrival.day_offset == 1
    arrival_at = arrival.arrival_at(SERVICE_DATE, TZ)
    assert arrival_at is not None
    assert arrival_at.date() == date(2026, 6, 16)
    # Localisation, not conversion: the published clock time is untouched.
    assert arrival_at.strftime("%H:%M") == "04:30"
    assert arrival_at.isoformat() == "2026-06-16T04:30:00+05:30"
    assert arrival.arrival_minutes == 24 * 60 + 4 * 60 + 30

    # A departure that itself crosses midnight is expressed as day_offset + 1.
    late = stop("NDLS", 4, arrival="23:50", departure="00:20", day_offset=1, departure_day_offset=2)
    assert late.departure_day_offset == 2
    assert late.departure_minutes > late.arrival_minutes


def test_decreasing_or_negative_day_offsets_are_rejected() -> None:
    with pytest.raises(DomainValidationError, match="must be >= 0"):
        stop("BLR", 1, departure="21:40", day_offset=-1)
    with pytest.raises(DomainValidationError, match="must never decrease"):
        Train(
            train_number="YT2006",
            train_name="Backwards In Time",
            stops=(
                stop("BLR", 1, departure="21:40", day_offset=1),
                stop("DEL", 2, arrival="06:00", day_offset=0),
            ),
        )


@pytest.mark.parametrize("offset", [0, 3, 5])
def test_departure_day_offset_must_be_day_offset_or_plus_one(offset: int) -> None:
    with pytest.raises(DomainValidationError, match="departure_day_offset must equal"):
        stop(
            "NDLS", 4, arrival="23:50", departure="00:20", day_offset=1,
            departure_day_offset=offset,
        )


def test_departure_day_offset_may_cross_midnight() -> None:
    late = stop("NDLS", 4, arrival="23:50", departure="00:20", day_offset=1, departure_day_offset=2)
    assert late.departure_day_offset == 2


def test_negative_departure_day_offset_is_rejected() -> None:
    with pytest.raises(DomainValidationError, match="must be >= 0"):
        stop("NDLS", 4, arrival="23:50", departure="00:20", day_offset=1, departure_day_offset=-1)


def test_arrival_departure_consistency_is_validated() -> None:
    with pytest.raises(DomainValidationError, match="must have an arrival time"):
        RailwayStationStop(station_code="BLR", sequence=1)
    with pytest.raises(DomainValidationError, match="precedes arrival"):
        stop("BLR", 1, arrival="21:40", departure="20:40")
    with pytest.raises(DomainValidationError, match="sub-minute precision"):
        RailwayStationStop(
            station_code="BLR", sequence=1, departure=time(21, 40, 30)
        )
    with pytest.raises(DomainValidationError, match="naive local time"):
        RailwayStationStop(
            station_code="BLR",
            sequence=1,
            departure=time(21, 40, tzinfo=UTC),
        )
    with pytest.raises(DomainValidationError, match="origin stop must have a departure"):
        Train(
            train_number="YT2007",
            train_name="No departure",
            stops=(stop("BLR", 1, arrival="21:40"), stop("DEL", 2, arrival="06:00", day_offset=1)),
        )
    with pytest.raises(DomainValidationError, match="terminus stop must have an arrival"):
        Train(
            train_number="YT2008",
            train_name="No arrival",
            stops=(
                stop("BLR", 1, departure="21:40"),
                stop("DEL", 2, departure="06:00", day_offset=1),
            ),
        )
    with pytest.raises(DomainValidationError, match="is not chronological"):
        Train(
            train_number="YT2009",
            train_name="Out of order",
            stops=(
                stop("BLR", 1, departure="10:00"),
                stop("HYD", 2, arrival="09:00", departure="09:30"),
                stop("DEL", 3, arrival="20:00"),
            ),
        )


# ----------------------------------------------------------------------
# route order
# ----------------------------------------------------------------------
def test_route_order_must_place_origin_before_destination() -> None:
    codes = ("BLR", "HYD", "DEL")
    assert require_ordered_route(codes, "BLR", "DEL") == (0, 2)
    with pytest.raises(DomainValidationError, match="never reversed"):
        require_ordered_route(codes, "DEL", "BLR")
    with pytest.raises(DomainValidationError, match="must differ"):
        require_ordered_route(codes, "BLR", "BLR")
    with pytest.raises(DomainValidationError, match="does not contain station"):
        require_ordered_route(codes, "BLR", "GOA")


# ----------------------------------------------------------------------
# travel class and quota
# ----------------------------------------------------------------------
def test_train_class_reuses_phase1_enum() -> None:
    assert TrainClass is TravelClass
    assert normalize_train_class("3a") is TravelClass.AC_3_TIER
    assert normalize_train_class(TravelClass.SLEEPER) is TravelClass.SLEEPER


def test_invalid_travel_class_is_rejected() -> None:
    with pytest.raises(DomainValidationError, match="unsupported travel class"):
        snapshot(travel_class="XX")
    with pytest.raises(DomainValidationError, match="must be a string"):
        snapshot(travel_class=3)


def test_quota_is_normalised_and_invalid_quota_is_rejected() -> None:
    assert coerce_quota(" tatkal ") is Quota.TATKAL
    assert coerce_quota("premium-tatkal") is Quota.PREMIUM_TATKAL
    assert Quota.GENERAL.value == "GENERAL"
    assert snapshot(quota="senior citizen").quota is Quota.SENIOR_CITIZEN
    with pytest.raises(DomainValidationError, match="unsupported quota"):
        coerce_quota("FRIENDS_AND_FAMILY")
    with pytest.raises(DomainValidationError, match="must be a string"):
        coerce_quota(7)
    with pytest.raises(DomainValidationError, match="unsupported quota"):
        snapshot(quota="NOT_A_QUOTA")


# ----------------------------------------------------------------------
# availability snapshots
# ----------------------------------------------------------------------
def test_valid_availability_snapshot_normalises_and_exposes_fare() -> None:
    snap = snapshot()
    assert snap.train_number == "YT1001"
    assert snap.origin_station_code == "BLR"
    assert snap.travel_class is TravelClass.AC_3_TIER
    assert snap.quota is Quota.GENERAL
    assert snap.state is AvailabilityState.AVAILABLE
    assert snap.is_confirmed is True
    assert snap.available_count == 42
    assert snap.key == ("YT1001", "BLR", "DEL", TRAVEL_DATE, "3A", "GENERAL")
    assert snap.fare == Fare.from_paise(245000)
    assert snap.fare is not None and str(snap.fare.rupees) == "2450"


def test_snapshot_rejects_identical_origin_and_destination() -> None:
    with pytest.raises(DomainValidationError, match="must differ"):
        snapshot(destination_station_code="blr")


def test_snapshot_state_counters_must_match_the_state() -> None:
    rac = snapshot(state=AvailabilityState.RAC, available_count=None, rac_count=12, fare_paise=None)
    assert rac.rac_count == 12
    waitlist = snapshot(
        state=AvailabilityState.WAITLIST, available_count=None, waitlist_value=7, fare_paise=None
    )
    assert waitlist.waitlist_value == 7

    with pytest.raises(DomainValidationError, match="reports state NOT_AVAILABLE"):
        snapshot(state=AvailabilityState.NOT_AVAILABLE, available_count=3, fare_paise=None)
    with pytest.raises(DomainValidationError, match="reports state RAC"):
        snapshot(state=AvailabilityState.RAC, available_count=3, rac_count=1, fare_paise=None)
    with pytest.raises(DomainValidationError, match="must be >= 1"):
        snapshot(available_count=0)


# ----------------------------------------------------------------------
# UNKNOWN handling
# ----------------------------------------------------------------------
def test_missing_state_defaults_to_unknown_and_is_never_confirmed() -> None:
    snap = AvailabilitySnapshot(
        train_number="YT1001",
        origin_station_code="BLR",
        destination_station_code="DEL",
        travel_date=TRAVEL_DATE,
        travel_class="3A",
        provider="unit-test-source",
        fetched_at=FETCHED_AT,
    )
    assert snap.state is AvailabilityState.UNKNOWN
    assert snap.is_confirmed is False
    assert snap.fare is None
    assert snap.available_count is None


def test_unreported_snapshot_stays_unknown() -> None:
    snap = AvailabilitySnapshot.unreported(
        train_number="YT1001",
        origin_station_code="BLR",
        destination_station_code="DEL",
        travel_date=TRAVEL_DATE,
        travel_class="3A",
        provider="unit-test-source",
        fetched_at=FETCHED_AT,
    )
    assert snap.state is AvailabilityState.UNKNOWN
    assert snap.is_confirmed is False
    assert snap.to_availability_record().is_confirmed is False  # type: ignore[attr-defined]


def test_unknown_snapshot_may_not_carry_an_availability_counter() -> None:
    with pytest.raises(DomainValidationError, match="reports state UNKNOWN"):
        snapshot(
            state=AvailabilityState.UNKNOWN,
            available_count=1,
            rac_count=None,
            fare_paise=None,
        )
    with pytest.raises(DomainValidationError, match="reports state UNKNOWN"):
        snapshot(
            state=AvailabilityState.UNKNOWN, available_count=None, rac_count=4, fare_paise=None
        )
    with pytest.raises(DomainValidationError, match="reports state UNKNOWN"):
        snapshot(
            state=AvailabilityState.UNKNOWN,
            available_count=None,
            waitlist_value=9,
            fare_paise=None,
        )


# ----------------------------------------------------------------------
# money
# ----------------------------------------------------------------------
def test_fare_is_paise_only_and_exact() -> None:
    assert snapshot(fare_paise=1).fare == Fare.from_paise(1)
    assert snapshot(fare_paise=0).fare == Fare.from_paise(0)
    assert snapshot(fare_paise=None).fare is None
    assert snapshot(fare_paise=245000).fare == Fare.from_paise(245000)
    assert snapshot(fare_paise=245000).to_json()["fare_paise"] == 245000
    assert require_paise(99, "fare_paise") == 99


@pytest.mark.parametrize(
    "bad", [2450.0, 2450.5, True, False, "245000", Decimal("245000"), object()]
)
def test_float_bool_or_non_integer_fare_is_rejected(bad: object) -> None:
    with pytest.raises(DomainValidationError):
        snapshot(fare_paise=bad)
    with pytest.raises(DomainValidationError):
        require_paise(bad, "fare_paise")




def test_negative_fare_is_rejected() -> None:
    with pytest.raises(DomainValidationError, match="must be >= 0"):
        snapshot(fare_paise=-1)
    with pytest.raises(DomainValidationError, match="must be >= 0"):
        require_paise(-1, "fare_paise")


# ----------------------------------------------------------------------
# provider metadata
# ----------------------------------------------------------------------
def test_provider_metadata_is_preserved_verbatim() -> None:
    snap = snapshot(provider="authorised-source-x", fetched_at=FETCHED_AT, updated_at=UPDATED_AT)
    assert snap.provider == "authorised-source-x"
    assert snap.fetched_at == FETCHED_AT
    assert snap.updated_at == UPDATED_AT
    body = snap.to_json()
    assert body["provider"] == "authorised-source-x"
    assert body["fetched_at"] == FETCHED_AT.isoformat()
    assert body["updated_at"] == UPDATED_AT.isoformat()


def test_metadata_is_not_invented_when_the_source_supplies_nothing() -> None:
    snap = snapshot(updated_at=None)
    assert snap.updated_at is None
    assert snap.to_json()["updated_at"] is None
    record = snap.to_availability_record()
    assert record.updated_at is None  # type: ignore[attr-defined]
    assert record.fetched_at is not None  # type: ignore[attr-defined]


def test_provenance_is_required_and_instants_must_be_timezone_aware() -> None:
    with pytest.raises(DomainValidationError, match="non-empty string"):
        snapshot(provider="")
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        snapshot(fetched_at=datetime(2026, 6, 30, 6, 15))
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        snapshot(updated_at=datetime(2026, 6, 30, 5, 45))
    with pytest.raises(DomainValidationError, match="later than fetched_at"):
        snapshot(fetched_at=UPDATED_AT, updated_at=FETCHED_AT)


# ----------------------------------------------------------------------
# bridging into the Phase 1 engine
# ----------------------------------------------------------------------
def test_snapshot_maps_to_a_phase1_provider_record_without_loss() -> None:
    record = snapshot().to_availability_record()
    assert record.train_number == "YT1001"  # type: ignore[attr-defined]
    assert record.state is AvailabilityState.AVAILABLE  # type: ignore[attr-defined]
    assert record.quota == "GENERAL"  # type: ignore[attr-defined]
    assert record.seats_available == 42  # type: ignore[attr-defined]
    assert record.fare == Fare.from_paise(245000)  # type: ignore[attr-defined]
    assert record.provider == "unit-test-source"  # type: ignore[attr-defined]
    # Phase 1 transports naive datetimes: the instant is preserved in UTC.
    assert record.fetched_at == FETCHED_AT.replace(tzinfo=None)  # type: ignore[attr-defined]
    assert record.updated_at == UPDATED_AT.replace(tzinfo=None)  # type: ignore[attr-defined]
    assert record.is_confirmed is True  # type: ignore[attr-defined]


class _ContractProvider:
    """A minimal provider adapter: contract snapshot in, Phase 1 record out."""

    def __init__(self, snapped: AvailabilitySnapshot | None) -> None:
        self._snapshot = snapped

    @property
    def provider_name(self) -> str:
        return "unit-test-source"

    def get_availability(self, query: object) -> object:
        if self._snapshot is None:
            return None
        return self._snapshot.to_availability_record()


def _segment_from_provider(snapped: AvailabilitySnapshot | None, state_expected: AvailabilityState):
    train = overnight_train().to_engine_train()
    timetable = Timetable(
        stations=[
            Station(code=code, name=code).to_engine_station() for code in ("BLR", "HYD", "DEL")
        ],
        trains=[train],
    )
    leg = RailLeg(train=train, origin_index=0, destination_index=2, anchor_date=TRAVEL_DATE)
    request = SearchRequest(
        origin="BLR", destination="DEL", travel_date=TRAVEL_DATE, travel_class="3A"
    )
    segment = build_segment(
        leg,
        request=request,
        availability_book=ProviderAvailabilityBook(_ContractProvider(snapped)),
        timetable=timetable,
        reservation_index=0,
    )
    assert segment.availability is state_expected
    return segment


def test_engine_consumes_a_contract_snapshot_through_the_provider_boundary() -> None:
    snapped = snapshot(travel_date=TRAVEL_DATE)
    segment = _segment_from_provider(snapped, AvailabilityState.AVAILABLE)
    assert segment.fare == Fare.from_paise(245000)
    assert segment.seats_available == 42
    assert segment.departure.isoformat() == "2026-07-01T21:40:00"


def test_a_provider_with_nothing_to_report_leaves_the_engine_segment_unknown() -> None:
    segment = _segment_from_provider(None, AvailabilityState.UNKNOWN)
    assert segment.fare is None
    assert segment.seats_available is None


def test_a_contract_unknown_snapshot_never_becomes_available() -> None:
    snapped = AvailabilitySnapshot.unreported(
        train_number="YT1001",
        origin_station_code="BLR",
        destination_station_code="DEL",
        travel_date=TRAVEL_DATE,
        travel_class="3A",
        provider="unit-test-source",
        fetched_at=FETCHED_AT,
    )
    segment = _segment_from_provider(snapped, AvailabilityState.UNKNOWN)
    assert segment.fare is None
