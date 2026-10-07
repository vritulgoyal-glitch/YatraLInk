"""Fixture loading and the shape of the synthetic snapshot."""

from __future__ import annotations

from datetime import date

import pytest

from engine.availability import Availability, DateAgnosticAvailability
from engine.enums import AvailabilityState
from engine.errors import DataSourceError
from engine.fixtures import (
    DEFAULT_FIXTURE_DIRECTORY,
    FIXTURE_FILES,
    default_network,
    iter_fixture_files,
    load_availability_book,
    load_fixture_network,
    load_stations,
    load_timetable,
    load_trains,
    load_transfer_allowances,
)
from engine.money import Fare
from engine.network import Timetable
from engine.search import SearchRequest
from engine.tests.builders import TRAVEL_DATE


class TestFixtureFiles:
    def test_all_declared_files_exist(self) -> None:
        for filename in FIXTURE_FILES.values():
            assert (DEFAULT_FIXTURE_DIRECTORY / filename).is_file(), filename

    def test_iter_fixture_files_is_deterministic(self) -> None:
        first = list(iter_fixture_files())
        second = list(iter_fixture_files())
        assert first == second
        assert len(first) == len(FIXTURE_FILES)

    def test_missing_directory_raises(self) -> None:
        with pytest.raises(DataSourceError):
            load_fixture_network("/nonexistent/fixture/directory")


class TestFixtureSnapshot:
    def test_trains_are_clearly_synthetic(self, network) -> None:
        """Every train number is prefixed YT so no real service can be implied."""
        assert all(train.number.startswith("YT") for train in network.timetable.trains)

    def test_stations_load(self, network) -> None:
        codes = network.timetable.station_codes
        assert {"BLR", "HYD", "DEL", "NDLS", "NGP", "AAA", "EEE"} <= set(codes)

    def test_every_station_used_by_a_train_is_declared(self, network) -> None:
        """Timetable construction would already fail, so this asserts the invariant."""
        declared = set(network.timetable.station_codes)
        for train in network.timetable.trains:
            assert set(train.station_codes) <= declared

    def test_scenarios_are_documented(self, network) -> None:
        assert len(network.scenarios) >= 10

    def test_transfer_allowance_is_declared(self, network) -> None:
        keys = {allowance.key for allowance in network.transfer_allowances}
        assert ("NDLS", "DEL") in keys

    def test_configuration_came_from_the_fixture(self, network) -> None:
        assert network.configuration.minimum_connection_minutes == 30
        assert network.configuration.allow_cross_station_transfers is True
        assert network.configuration.same_train_split_max_segments == 3

    def test_stations_sorted_deterministically(self, network) -> None:
        codes = network.timetable.station_codes
        assert list(codes) == sorted(codes)

    def test_default_network_is_cached(self) -> None:
        assert default_network() is default_network()

    def test_with_configuration_keeps_the_snapshot(self, network) -> None:
        from engine.config import SearchConfiguration

        modified = network.with_configuration(SearchConfiguration(enable_connecting=False))
        assert modified.timetable is network.timetable
        assert modified.availability_book is network.availability_book
        assert modified.configuration.enable_connecting is False


class TestFixtureInterpretation:
    def test_multi_day_route_has_day_offsets_up_to_two(self, network) -> None:
        train = network.timetable.train("YT1013")
        offsets = [stop.day_offset for stop in train.stops]
        assert max(offsets) == 2
        assert offsets == [0, 1, 2, 2]

    def test_overnight_route_is_represented_with_an_offset(self, network) -> None:
        train = network.timetable.train("YT1001")
        assert [stop.day_offset for stop in train.stops] == [0, 1, 1]

    def test_availability_states_cover_every_documented_state(self, network) -> None:
        states = {entry.state for entry in network.availability_book.entries}
        assert states == {
            AvailabilityState.AVAILABLE,
            AvailabilityState.RAC,
            AvailabilityState.WAITLIST,
            AvailabilityState.NOT_AVAILABLE,
        }
        # UNKNOWN is represented by *absence* of a record, never by a stored value.
        assert AvailabilityState.UNKNOWN not in states

    def test_segment_without_a_record_is_unknown_not_invented(self, network) -> None:
        """YT1014 HYD->NDLS is deliberately absent from the snapshot."""
        record = network.availability_book.lookup(
            train_number="YT1014",
            origin_station_code="HYD",
            destination_station_code="NDLS",
            travel_date=TRAVEL_DATE,
            travel_class="3A",
        )
        assert record is None

    def test_fares_are_integer_paise(self, network) -> None:
        for entry in network.availability_book.entries:
            if entry.fare is not None:
                assert isinstance(entry.fare, Fare)
                assert isinstance(entry.fare.amount_paise, int)

    def test_rac_and_waitlist_records_exist(self, network) -> None:
        by_state: dict[AvailabilityState, int] = {}
        for entry in network.availability_book.entries:
            by_state[entry.state] = by_state.get(entry.state, 0) + 1
        assert by_state[AvailabilityState.RAC] >= 1
        assert by_state[AvailabilityState.WAITLIST] >= 1
        assert by_state[AvailabilityState.NOT_AVAILABLE] >= 1

    def test_lowest_fare_is_cheaper_than_highest_fare(self, network) -> None:
        fares = [
            entry.fare.amount_paise
            for entry in network.availability_book.entries
            if entry.fare is not None
        ]
        assert min(fares) < max(fares)


class TestLoaderErrors:
    def test_timetable_rejects_a_train_at_an_undeclared_station(self, tmp_path) -> None:
        stations = tmp_path / "stations.json"
        stations.write_text('{"stations": [{"code": "AAA", "name": "Alpha"}]}', encoding="utf-8")
        trains = tmp_path / "trains.json"
        trains.write_text(
            """
            {"trains": [{"number": "YT1", "name": "X", "stops": [
                {"station_code": "AAA", "sequence": 1, "departure": "06:00", "day_offset": 0},
                {"station_code": "ZZZ", "sequence": 2, "arrival": "08:00", "day_offset": 0}
            ]}]}
            """,
            encoding="utf-8",
        )
        with pytest.raises(DataSourceError):
            load_timetable(trains, stations)

    def test_duplicate_station_code_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "stations.json"
        path.write_text(
            '{"stations": [{"code": "AAA", "name": "A"}, {"code": "aaa", "name": "A2"}]}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            from engine.models import Station

            Timetable(
                stations=(
                    Station(code="AAA", name="A"),
                    Station(code="AAA", name="A2"),
                ),
                trains=(),
            )

    def test_missing_key_is_reported(self, tmp_path) -> None:
        path = tmp_path / "stations.json"
        path.write_text('{"stations": [{"code": "AAA"}]}', encoding="utf-8")
        with pytest.raises(DataSourceError):
            load_stations(path)

    def test_unknown_configuration_key_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "network.json"
        path.write_text(
            """
            {"name": "x", "description": "y", "scenarios": [],
             "search_configuration": {"minimum_connection_minutes": 30, "typo_key": 1}}
            """,
            encoding="utf-8",
        )
        from engine.fixtures import _read_json, load_search_configuration

        with pytest.raises(ValueError):
            load_search_configuration(_read_json(path).get("search_configuration"))

    def test_unknown_ranking_weight_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "network.json"
        path.write_text(
            """
            {"name": "x", "description": "y", "scenarios": [],
             "search_configuration": {"ranking_weights": {"availabilityy": 10}}}
            """,
            encoding="utf-8",
        )
        from engine.fixtures import _read_json, load_search_configuration

        with pytest.raises(ValueError):
            load_search_configuration(_read_json(path).get("search_configuration"))

    def test_nested_network_directory_loads(self, tmp_path) -> None:
        """A caller-supplied directory is honoured, proving fixtures are pluggable."""
        (tmp_path / "network.json").write_text(
            '{"name": "tiny", "description": "one train", "scenarios": []}', encoding="utf-8"
        )
        (tmp_path / "stations.json").write_text(
            '{"stations": [{"code": "AAA", "name": "A"}, {"code": "BBB", "name": "B"}]}',
            encoding="utf-8",
        )
        (tmp_path / "trains.json").write_text(
            """
            {"trains": [{"number": "YT1", "name": "Tiny", "stops": [
                {"station_code": "AAA", "sequence": 1, "departure": "06:00", "day_offset": 0},
                {"station_code": "BBB", "sequence": 2, "arrival": "08:00", "day_offset": 0}
            ]}]}
            """,
            encoding="utf-8",
        )
        (tmp_path / "availability.json").write_text(
            """
            {"availability": [{"train_number": "YT1", "origin_station_code": "AAA",
             "destination_station_code": "BBB", "travel_class": "3A",
             "state": "AVAILABLE", "fare_paise": 1000, "travel_date": null}]}
            """,
            encoding="utf-8",
        )
        (tmp_path / "transfer_allowances.json").write_text(
            '{"transfer_allowances": []}', encoding="utf-8"
        )

        loaded = load_fixture_network(tmp_path)
        assert loaded.name == "tiny"
        assert len(loaded.timetable.trains) == 1
        assert loaded.configuration.minimum_connection_minutes == 30

        from engine.pipeline import generate_journeys

        result = generate_journeys(
            SearchRequest(origin="AAA", destination="BBB", travel_date=date(2026, 7, 1)),
            timetable=loaded.timetable,
            availability_book=loaded.availability_book,
            configuration=loaded.configuration,
        )
        assert len(result.journeys) == 1
        assert result.best.total_fare == Fare.from_paise(1000)


class TestFixtureLoaders:
    def test_load_stations_returns_station_models(self) -> None:
        stations = load_stations(DEFAULT_FIXTURE_DIRECTORY / FIXTURE_FILES["stations"])
        assert stations
        assert all(station.code == station.code.upper() for station in stations)

    def test_load_trains_returns_route_models(self) -> None:
        trains = load_trains(DEFAULT_FIXTURE_DIRECTORY / FIXTURE_FILES["trains"])
        assert trains
        assert all(len(train.stops) >= 2 for train in trains)

    def test_load_availability_book_marks_date_independent_entries(self) -> None:
        book = load_availability_book(DEFAULT_FIXTURE_DIRECTORY / FIXTURE_FILES["availability"])
        assert all(isinstance(entry, DateAgnosticAvailability) for entry in book.entries)

    def test_dated_entry_takes_precedence_over_date_independent(self) -> None:
        from engine.availability import AvailabilityBook

        book = AvailabilityBook(
            [
                DateAgnosticAvailability(
                    train_number="YT1",
                    origin_station_code="AAA",
                    destination_station_code="BBB",
                    travel_class="3A",
                    state=AvailabilityState.AVAILABLE,
                    fare=Fare.from_paise(1000),
                ),
                Availability(
                    train_number="YT1",
                    origin_station_code="AAA",
                    destination_station_code="BBB",
                    travel_date=TRAVEL_DATE,
                    travel_class="3A",
                    state=AvailabilityState.WAITLIST,
                    fare=Fare.from_paise(2000),
                ),
            ]
        )
        assert (
            book.lookup(
                train_number="YT1",
                origin_station_code="AAA",
                destination_station_code="BBB",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
            ).state
            is AvailabilityState.WAITLIST
        )
        assert (
            book.lookup(
                train_number="YT1",
                origin_station_code="AAA",
                destination_station_code="BBB",
                travel_date=date(2027, 1, 1),
                travel_class="3A",
            ).state
            is AvailabilityState.AVAILABLE
        )

    def test_duplicate_availability_entry_is_rejected(self) -> None:
        from engine.availability import AvailabilityBook

        entry = DateAgnosticAvailability(
            train_number="YT1",
            origin_station_code="AAA",
            destination_station_code="BBB",
            travel_class="3A",
            state=AvailabilityState.AVAILABLE,
        )
        with pytest.raises(ValueError):
            AvailabilityBook([entry, entry])

    def test_load_transfer_allowances(self) -> None:
        allowances = load_transfer_allowances(
            DEFAULT_FIXTURE_DIRECTORY / FIXTURE_FILES["transfer_allowances"]
        )
        assert allowances
        assert allowances[0].minimum_minutes >= 1

    def test_allowance_between_identical_stations_is_rejected(self) -> None:
        from engine.network import TransferAllowance

        with pytest.raises(ValueError):
            TransferAllowance(from_station_code="DEL", to_station_code="del", minimum_minutes=30)
