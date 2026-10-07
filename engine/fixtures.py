"""Loading synthetic railway fixtures into engine structures.

The fixture layer is deliberately the *only* place that knows about files.  It
turns JSON into a :class:`FixtureNetwork` — a :class:`engine.network.Timetable`,
an :class:`engine.availability.AvailabilityBook`, explicit transfer allowances
and a :class:`engine.config.SearchConfiguration`.

Because the engine's public entry point takes those four objects, a future
railway-data adapter can replace this module (or this module's JSON) without any
algorithm in the engine changing.  Nothing here performs network I/O: it reads
local files only.

The shipped fixtures are **synthetic sample data**.  They are not Indian Railways
schedules, and the train numbers are deliberately prefixed ``YT`` to make that
unambiguous.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.availability import Availability, AvailabilityBook, DateAgnosticAvailability
from engine.config import RankingWeights, SearchConfiguration
from engine.errors import DataSourceError, DomainValidationError
from engine.models import Station, Train, TrainStop
from engine.money import Fare
from engine.network import Timetable, TransferAllowance
from engine.search import SearchRequest
from engine.time_utils import parse_clock

__all__ = [
    "DEFAULT_FIXTURE_DIRECTORY",
    "FIXTURE_FILES",
    "FixtureNetwork",
    "default_network",
    "load_availability_book",
    "load_fixture_network",
    "load_stations",
    "load_timetable",
    "load_trains",
    "load_transfer_allowances",
]

#: Repository-relative location of the shipped synthetic fixtures.
DEFAULT_FIXTURE_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "fixtures"

FIXTURE_FILES: Mapping[str, str] = {
    "network": "network.json",
    "stations": "stations.json",
    "trains": "trains.json",
    "availability": "availability.json",
    "transfer_allowances": "transfer_allowances.json",
}


@dataclass(frozen=True, slots=True)
class FixtureNetwork:
    """A complete, self-contained railway snapshot the engine can be run against."""

    name: str
    description: str
    scenarios: tuple[str, ...]
    timetable: Timetable
    availability_book: AvailabilityBook
    transfer_allowances: tuple[TransferAllowance, ...]
    configuration: SearchConfiguration

    def with_configuration(self, configuration: SearchConfiguration) -> FixtureNetwork:
        """Copy of this network using a different engine configuration."""
        return FixtureNetwork(
            name=self.name,
            description=self.description,
            scenarios=self.scenarios,
            timetable=self.timetable,
            availability_book=self.availability_book,
            transfer_allowances=self.transfer_allowances,
            configuration=configuration,
        )


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise DataSourceError(f"fixture file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise DataSourceError(f"fixture file {path} is not valid JSON: {exc}") from exc


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataSourceError(f"{label} must be a JSON object, got {type(value).__name__}")
    return value


def _require_list(mapping: Mapping[str, Any], key: str, label: str) -> list[Any]:
    if key not in mapping:
        raise DataSourceError(f"{label} is missing the required key {key!r}")
    value = mapping[key]
    if not isinstance(value, list):
        raise DataSourceError(f"{label}.{key} must be a JSON array, got {type(value).__name__}")
    return value


def load_stations(path: Path | str) -> tuple[Station, ...]:
    """Load stations from a JSON file."""
    document = _require_mapping(_read_json(Path(path)), "stations document")
    stations = []
    for entry in _require_list(document, "stations", "stations document"):
        item = _require_mapping(entry, "station entry")
        try:
            stations.append(Station(code=item["code"], name=item["name"]))
        except KeyError as exc:
            raise DataSourceError(f"station entry is missing key {exc.args[0]!r}") from exc
    return tuple(stations)


def load_trains(path: Path | str) -> tuple[Train, ...]:
    """Load train routes from a JSON file."""
    document = _require_mapping(_read_json(Path(path)), "trains document")
    trains: list[Train] = []
    for entry in _require_list(document, "trains", "trains document"):
        item = _require_mapping(entry, "train entry")
        try:
            number = item["number"]
            name = item["name"]
            raw_stops = item["stops"]
        except KeyError as exc:
            raise DataSourceError(f"train entry is missing key {exc.args[0]!r}") from exc
        if not isinstance(raw_stops, list) or not raw_stops:
            raise DataSourceError(f"train {number!r} must declare a non-empty stops array")
        stops = []
        for raw_stop in raw_stops:
            stop = _require_mapping(raw_stop, f"stop of train {number!r}")
            arrival = stop.get("arrival")
            departure = stop.get("departure")
            stops.append(
                TrainStop(
                    station_code=stop["station_code"],
                    sequence=stop["sequence"],
                    arrival=parse_clock(arrival) if arrival else None,
                    departure=parse_clock(departure) if departure else None,
                    day_offset=stop.get("day_offset", 0),
                    departure_day_offset=stop.get("departure_day_offset"),
                )
            )
        trains.append(Train(number=number, name=name, stops=tuple(stops)))
    return tuple(trains)


def load_timetable(
    trains_path: Path | str,
    stations_path: Path | str,
    transfer_allowances: Iterable[TransferAllowance] = (),
) -> Timetable:
    """Load a :class:`engine.network.Timetable` from train and station fixtures.

    ``transfer_allowances`` travel with the timetable because they are part of
    the same data snapshot: where the data declares that two distinct stations
    may be used to change trains.
    """
    return Timetable(
        stations=load_stations(stations_path),
        trains=load_trains(trains_path),
        transfer_allowances=transfer_allowances,
    )


def load_availability_book(path: Path | str) -> AvailabilityBook:
    """Load an :class:`engine.availability.AvailabilityBook` from a JSON file.

    An entry with ``"travel_date": null`` becomes a
    :class:`engine.availability.DateAgnosticAvailability`, which is how the
    synthetic snapshot declares "this service reports the same state on any date".
    """
    document = _require_mapping(_read_json(Path(path)), "availability document")
    entries: list[Availability | DateAgnosticAvailability] = []
    for entry in _require_list(document, "availability", "availability document"):
        item = _require_mapping(entry, "availability entry")
        fare_paise = item.get("fare_paise")
        fare = Fare.from_paise(fare_paise) if fare_paise is not None else None
        travel_date = item.get("travel_date")
        common = {
            "train_number": item["train_number"],
            "origin_station_code": item["origin_station_code"],
            "destination_station_code": item["destination_station_code"],
            "travel_class": item["travel_class"],
            "state": item["state"],
            "quota": item.get("quota", "GENERAL"),
            "seats_available": item.get("seats_available"),
            "fare": fare,
        }
        if travel_date is None:
            entries.append(DateAgnosticAvailability(**common))
        else:
            parsed = date.fromisoformat(str(travel_date))
            entries.append(Availability(travel_date=parsed, **common))
    return AvailabilityBook(entries)


def load_transfer_allowances(path: Path | str) -> tuple[TransferAllowance, ...]:
    """Load explicit different-station transfer allowances."""
    document = _require_mapping(_read_json(Path(path)), "transfer allowance document")
    allowances = []
    for entry in _require_list(document, "transfer_allowances", "transfer allowance document"):
        item = _require_mapping(entry, "transfer allowance entry")
        allowances.append(
            TransferAllowance(
                from_station_code=item["from_station_code"],
                to_station_code=item["to_station_code"],
                minimum_minutes=item["minimum_minutes"],
                note=item.get("note", ""),
            )
        )
    return tuple(allowances)


def load_search_configuration(document: Mapping[str, Any] | None) -> SearchConfiguration:
    """Build a :class:`engine.config.SearchConfiguration` from a fixture document.

    Unknown keys are rejected rather than ignored, so a typo in a fixture cannot
    silently change engine behaviour.
    """
    if document is None:
        return SearchConfiguration()
    payload = dict(document)
    weights_document = payload.pop("ranking_weights", None)
    if weights_document is not None:
        weights_map = dict(_require_mapping(weights_document, "ranking_weights"))
        valid_weight_keys = set(RankingWeights.__dataclass_fields__)
        unknown = set(weights_map) - valid_weight_keys
        if unknown:
            raise DomainValidationError(f"unknown ranking weight(s) in fixture: {sorted(unknown)}")
        payload["ranking_weights"] = RankingWeights(**weights_map)
    valid_keys = set(SearchConfiguration.__dataclass_fields__)
    unknown_keys = set(payload) - valid_keys
    if unknown_keys:
        raise DomainValidationError(
            f"unknown search configuration key(s) in fixture: {sorted(unknown_keys)}"
        )
    return SearchConfiguration(**payload)


def load_fixture_network(directory: Path | str | None = None) -> FixtureNetwork:
    """Load a complete fixture network from a directory of JSON files."""
    base = Path(directory) if directory is not None else DEFAULT_FIXTURE_DIRECTORY
    network_document = _require_mapping(
        _read_json(base / FIXTURE_FILES["network"]), "network document"
    )
    allowances = load_transfer_allowances(base / FIXTURE_FILES["transfer_allowances"])
    timetable = load_timetable(
        trains_path=base / FIXTURE_FILES["trains"],
        stations_path=base / FIXTURE_FILES["stations"],
        transfer_allowances=allowances,
    )
    availability_book = load_availability_book(base / FIXTURE_FILES["availability"])
    configuration = load_search_configuration(network_document.get("search_configuration"))
    scenarios = network_document.get("scenarios", [])
    if not isinstance(scenarios, list) or not all(isinstance(s, str) for s in scenarios):
        raise DataSourceError("network.scenarios must be an array of strings")
    return FixtureNetwork(
        name=str(network_document.get("name", base.name)),
        description=str(network_document.get("description", "")),
        scenarios=tuple(scenarios),
        timetable=timetable,
        availability_book=availability_book,
        transfer_allowances=allowances,
        configuration=configuration,
    )


@lru_cache(maxsize=1)
def default_network() -> FixtureNetwork:
    """The shipped synthetic fixture network.

    Cached because the fixture snapshot is immutable; call
    :func:`load_fixture_network` directly for a fresh instance or a different
    directory.
    """
    return load_fixture_network()


def generate_journey_options(
    search_request: SearchRequest,
    fixture_network: FixtureNetwork,
    configuration: SearchConfiguration | None = None,
    limit: int | None = None,
) -> tuple:  # type: ignore[type-arg]
    """Generate ranked journeys against a fixture network.

    Thin, readable entry point for callers that already hold a
    :class:`FixtureNetwork`.  The heavy lifting lives in
    :func:`engine.pipeline.generate_journeys`.
    """
    from engine.pipeline import generate_journeys

    result = generate_journeys(
        search_request,
        timetable=fixture_network.timetable,
        availability_book=fixture_network.availability_book,
        configuration=configuration or fixture_network.configuration,
        transfer_allowances=fixture_network.transfer_allowances,
        limit=limit,
    )
    return result.journeys


def iter_fixture_files(directory: Path | str | None = None) -> Iterable[Path]:
    """Yield the fixture files of a directory in a deterministic order."""
    base = Path(directory) if directory is not None else DEFAULT_FIXTURE_DIRECTORY
    for key in sorted(FIXTURE_FILES):
        yield base / FIXTURE_FILES[key]
