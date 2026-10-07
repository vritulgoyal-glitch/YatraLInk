"""The Phase 2B development timetable fixture provider.

Focused tests for :mod:`engine.railway_data.providers.timetable_fixture`,
proving the data path::

    railway timetable data (development fixture JSON)
        -> provider adapter
        -> Phase 2A normalized models
        -> Phase 1 engine types

Nothing here touches availability, fares, quotas, IRCTC or the network: the
fixture is development data and the tests assert that it never claims
otherwise.
"""

from __future__ import annotations

import json
from datetime import date, time
from typing import Any

import pytest

from engine import time_utils as tu
from engine.errors import DataSourceError, DomainValidationError
from engine.models import normalize_train_number
from engine.railway_data import RailwayStationStop, Station, Train, TrainSchedule
from engine.railway_data.providers import TimetableFixtureProvider

SERVICE_DATE = date(2026, 10, 7)
OTHER_DATE = date(2026, 12, 25)


@pytest.fixture()
def provider() -> TimetableFixtureProvider:
    """The shipped development timetable fixture provider."""
    return TimetableFixtureProvider()


# ----------------------------------------------------------------------
# 1. station lookup
# ----------------------------------------------------------------------
def test_get_station_returns_normalized_contract_station(
    provider: TimetableFixtureProvider,
) -> None:
    station = provider.get_station("BLR")
    assert isinstance(station, Station)
    assert station.code == "BLR"
    assert station.name == "Dev Fixture Bengaluru City Junction"
    assert station.city == "Bengaluru"
    assert station.timezone == "Asia/Kolkata"


def test_get_station_normalizes_case_and_whitespace(provider: TimetableFixtureProvider) -> None:
    station = provider.get_station("  blr ")
    assert station is not None
    assert station.code == "BLR"


def test_get_station_unknown_returns_none(provider: TimetableFixtureProvider) -> None:
    assert provider.get_station("ZZZ") is None


# ----------------------------------------------------------------------
# 2. train lookup
# ----------------------------------------------------------------------
def test_get_train_returns_contract_train_with_ordered_route(
    provider: TimetableFixtureProvider,
) -> None:
    train = provider.get_train("ytf2201")
    assert isinstance(train, Train)
    assert train.train_number == normalize_train_number(" ytf2201 ")
    assert train.train_number == "YTF2201"
    assert train.station_codes == ("BLR", "GTL", "HYD", "NGP", "DEL")


def test_get_train_unknown_returns_none(provider: TimetableFixtureProvider) -> None:
    assert provider.get_train("YTF9999") is None


# ----------------------------------------------------------------------
# 3. full schedule lookup
# ----------------------------------------------------------------------
def test_get_schedule_returns_normalized_schedule(provider: TimetableFixtureProvider) -> None:
    schedule = provider.get_schedule("YTF2201", SERVICE_DATE)
    assert isinstance(schedule, TrainSchedule)
    assert schedule.train_number == "YTF2201"
    assert schedule.service_date == SERVICE_DATE
    assert isinstance(schedule.resolved_stops[0], RailwayStationStop)


def test_get_schedule_unknown_train_returns_none_and_creates_nothing(
    provider: TimetableFixtureProvider,
) -> None:
    assert provider.get_schedule("YTF0000", SERVICE_DATE) is None


# ----------------------------------------------------------------------
# 4. BLR -> HYD route lookup (direct journey scenario)
# ----------------------------------------------------------------------
def test_trains_between_blr_and_hyd(provider: TimetableFixtureProvider) -> None:
    result = provider.trains_between_stations("BLR", "HYD", SERVICE_DATE)
    assert result.origin_station_code == "BLR"
    assert result.destination_station_code == "HYD"
    numbers = [train.train_number for train in result.trains]
    assert numbers == ["YTF2201", "YTF2203"]  # deterministic: departure, then number


# ----------------------------------------------------------------------
# 5. BLR -> DEL route lookup (multi-hop route)
# ----------------------------------------------------------------------
def test_trains_between_blr_and_del(provider: TimetableFixtureProvider) -> None:
    result = provider.trains_between_stations("BLR", "DEL", SERVICE_DATE)
    numbers = [train.train_number for train in result.trains]
    assert numbers == ["YTF2201"]


# ----------------------------------------------------------------------
# 6. invalid / reversed route
# ----------------------------------------------------------------------
def test_reversed_route_returns_no_trains_and_is_never_fixed(
    provider: TimetableFixtureProvider,
) -> None:
    result = provider.trains_between_stations("HYD", "BLR", SERVICE_DATE)
    assert result.trains == ()


def test_equal_origin_and_destination_returns_no_trains(
    provider: TimetableFixtureProvider,
) -> None:
    result = provider.trains_between_stations("BLR", "blr", SERVICE_DATE)
    assert result.trains == ()


# ----------------------------------------------------------------------
# 7. unknown station in route query
# ----------------------------------------------------------------------
def test_route_query_with_unknown_station_is_empty(provider: TimetableFixtureProvider) -> None:
    assert provider.trains_between_stations("BLR", "ZZZ", SERVICE_DATE).trains == ()
    assert provider.trains_between_stations("ZZZ", "DEL", SERVICE_DATE).trains == ()


# ----------------------------------------------------------------------
# 8. unknown train (schedule and train lookups)
# ----------------------------------------------------------------------
def test_unknown_train_behaviour_is_consistent(provider: TimetableFixtureProvider) -> None:
    assert provider.get_train("YTF0000") is None
    assert provider.get_schedule("YTF0000", SERVICE_DATE) is None


# ----------------------------------------------------------------------
# 9. overnight day_offset (explicit, never inferred)
# ----------------------------------------------------------------------
def test_overnight_arrival_keeps_explicit_day_offset(
    provider: TimetableFixtureProvider,
) -> None:
    schedule = provider.get_schedule("YTF2201", SERVICE_DATE)
    assert schedule is not None
    del_stop = schedule.stop_for("DEL")
    assert del_stop.arrival == time(11, 0)
    assert del_stop.day_offset == 1  # explicit in the data, not derived from the clock
    blr_stop = schedule.stop_for("BLR")
    assert blr_stop.day_offset == 0
    assert blr_stop.departure == time(6, 0)


def test_midnight_crossing_halt_keeps_departure_day_offset(
    provider: TimetableFixtureProvider,
) -> None:
    schedule = provider.get_schedule("YTF2202", SERVICE_DATE)
    assert schedule is not None
    ngp_stop = schedule.stop_for("NGP")
    assert ngp_stop.arrival == time(23, 50)
    assert ngp_stop.day_offset == 0
    assert ngp_stop.departure == time(0, 20)
    assert ngp_stop.departure_day_offset == 1


def test_service_date_preserved_verbatim_on_schedule_and_result(
    provider: TimetableFixtureProvider,
) -> None:
    schedule = provider.get_schedule("YTF2202", OTHER_DATE)
    assert schedule is not None
    assert schedule.service_date == OTHER_DATE
    assert schedule.service_date_for(schedule.stop_for("DEL")) == date(2026, 12, 26)
    result = provider.trains_between_stations("BLR", "DEL", OTHER_DATE)
    assert result.service_date == OTHER_DATE


# ----------------------------------------------------------------------
# 10. intermediate station ordering
# ----------------------------------------------------------------------
def test_intermediate_stations_are_ordered_by_sequence(
    provider: TimetableFixtureProvider,
) -> None:
    train = provider.get_train("YTF2201")
    assert train is not None
    segment = train.stops_between("BLR", "HYD")
    assert [stop.station_code for stop in segment] == ["BLR", "GTL", "HYD"]
    sequences = [stop.sequence for stop in segment]
    assert sequences == sorted(sequences)


# ----------------------------------------------------------------------
# 11. deterministic result ordering
# ----------------------------------------------------------------------
def test_route_results_are_deterministic_across_instances() -> None:
    first = TimetableFixtureProvider().trains_between_stations("BLR", "DEL", SERVICE_DATE)
    second = TimetableFixtureProvider().trains_between_stations("BLR", "DEL", SERVICE_DATE)
    assert first.trains == second.trains
    keys = [(t.stop_for("BLR").effective_departure_minutes, t.train_number) for t in first.trains]
    assert keys == sorted(keys)


# ----------------------------------------------------------------------
# 12. multiple trains between stations
# ----------------------------------------------------------------------
def test_multiple_trains_between_same_pair_ordered_by_departure(
    provider: TimetableFixtureProvider,
) -> None:
    result = provider.trains_between_stations("BLR", "HYD", SERVICE_DATE)
    assert len(result.trains) == 2
    departures = [train.stop_for("BLR").departure for train in result.trains]
    assert departures == sorted(departures, key=lambda clock: (clock.hour, clock.minute))


# ----------------------------------------------------------------------
# 13. service-date preservation (schedule identity is date-independent)
# ----------------------------------------------------------------------
def test_schedule_route_is_independent_of_service_date(
    provider: TimetableFixtureProvider,
) -> None:
    first = provider.get_schedule("YTF2201", SERVICE_DATE)
    second = provider.get_schedule("YTF2201", OTHER_DATE)
    assert first is not None and second is not None
    assert first.resolved_stops == second.resolved_stops
    assert first.service_date != second.service_date


# ----------------------------------------------------------------------
# 14. Phase 2A validation integration (no duplicated validation)
# ----------------------------------------------------------------------
def _write_fixture(tmp_path: Any, document: dict[str, Any]) -> Any:
    target = tmp_path / "timetable.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


BASE_DOCUMENT: dict[str, Any] = {
    "metadata": {
        "data_type": "DEVELOPMENT_FIXTURE_DATA",
        "is_live_data": False,
    },
    "stations": [{"code": "BLR", "name": "Dev Fixture Bengaluru"}],
    "trains": [
        {
            "train_number": "YTF9001",
            "train_name": "Dev Fixture Validation Express",
            "stops": [
                {"station_code": "BLR", "sequence": 1, "arrival": None, "departure": "08:00"},
                {"station_code": "HYD", "sequence": 2, "arrival": "14:00", "departure": None},
            ],
        }
    ],
}


def test_duplicate_stop_sequence_is_rejected_by_phase_2a_validation(tmp_path: Any) -> None:
    document = json.loads(json.dumps(BASE_DOCUMENT))
    document["trains"][0]["stops"][1]["sequence"] = 1  # duplicate sequence 1
    path = _write_fixture(tmp_path, document)
    with pytest.raises(DomainValidationError, match="must be unique"):
        TimetableFixtureProvider(path=path)


def test_non_chronological_route_is_rejected_by_phase_2a_validation(tmp_path: Any) -> None:
    document = json.loads(json.dumps(BASE_DOCUMENT))
    document["trains"][0]["stops"][1]["arrival"] = "07:00"  # before the 08:00 departure
    path = _write_fixture(tmp_path, document)
    with pytest.raises(DomainValidationError, match="not chronological"):
        TimetableFixtureProvider(path=path)


def test_invalid_station_code_is_rejected_by_phase_2a_validation(tmp_path: Any) -> None:
    document = json.loads(json.dumps(BASE_DOCUMENT))
    document["stations"][0]["code"] = "1BAD"
    path = _write_fixture(tmp_path, document)
    with pytest.raises(DomainValidationError, match="station code"):
        TimetableFixtureProvider(path=path)


def test_missing_terminus_arrival_is_rejected_by_phase_2a_validation(tmp_path: Any) -> None:
    document = json.loads(json.dumps(BASE_DOCUMENT))
    document["trains"][0]["stops"][-1]["arrival"] = None
    path = _write_fixture(tmp_path, document)
    with pytest.raises(DomainValidationError, match="must have an arrival time"):
        TimetableFixtureProvider(path=path)


def test_bad_day_offset_is_rejected_by_phase_2a_validation(tmp_path: Any) -> None:
    document = json.loads(json.dumps(BASE_DOCUMENT))
    document["trains"][0]["stops"][1]["day_offset"] = -1
    path = _write_fixture(tmp_path, document)
    with pytest.raises(DomainValidationError, match="day_offset"):
        TimetableFixtureProvider(path=path)


def test_unmarked_or_live_marked_fixture_is_refused(tmp_path: Any) -> None:
    unmarked = json.loads(json.dumps(BASE_DOCUMENT))
    del unmarked["metadata"]
    path = _write_fixture(tmp_path, unmarked)
    with pytest.raises(DataSourceError, match="development fixture"):
        TimetableFixtureProvider(path=path)

    live = json.loads(json.dumps(BASE_DOCUMENT))
    live["metadata"]["is_live_data"] = True
    path = _write_fixture(tmp_path, live)
    with pytest.raises(DataSourceError, match="development fixture"):
        TimetableFixtureProvider(path=path)


def test_duplicate_station_or_train_in_fixture_is_refused(tmp_path: Any) -> None:
    document = json.loads(json.dumps(BASE_DOCUMENT))
    document["stations"].append({"code": "BLR", "name": "Dev Fixture Duplicate"})
    path = _write_fixture(tmp_path, document)
    with pytest.raises(DataSourceError, match="station BLR twice"):
        TimetableFixtureProvider(path=path)

    document = json.loads(json.dumps(BASE_DOCUMENT))
    document["trains"].append(document["trains"][0])
    path = _write_fixture(tmp_path, document)
    with pytest.raises(DataSourceError, match="train YTF9001 twice"):
        TimetableFixtureProvider(path=path)


# ----------------------------------------------------------------------
# 15. the fixture never claims live availability
# ----------------------------------------------------------------------
def test_fixture_never_claims_live_availability(provider: TimetableFixtureProvider) -> None:
    metadata = provider.fixture_metadata
    assert metadata["data_type"] == "DEVELOPMENT_FIXTURE_DATA"
    assert metadata["is_live_data"] is False
    assert metadata["is_irctc_data"] is False
    assert provider.is_development_fixture is True
    assert provider.provider_name == "development-timetable-fixture"
    with provider._fixture_path.open("r", encoding="utf-8") as handle:  # noqa: SLF001
        raw = handle.read()
    assert "availability" not in json.dumps(
        json.loads(raw)["trains"]
    )


def test_adapter_bridges_into_phase_1_engine_types(provider: TimetableFixtureProvider) -> None:
    """The contract models bridge losslessly into Phase 1's own types."""
    train = provider.get_train("YTF2201")
    assert train is not None
    engine_train = train.to_engine_train()
    assert engine_train.number == "YTF2201"
    assert tuple(stop.station_code for stop in engine_train.stops) == (
        "BLR",
        "GTL",
        "HYD",
        "NGP",
        "DEL",
    )
    first_departure = engine_train.stops[0].departure
    assert first_departure is not None
    assert tu.format_clock(first_departure) == "06:00"
