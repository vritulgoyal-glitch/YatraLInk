"""The development timetable fixture provider (Phase 2B).

Proves the Phase 2B data path end to end::

    railway timetable data (development fixture JSON)
        -> provider adapter (this module)
        -> Phase 2A normalized models (engine.railway_data)
        -> Phase 1 journey engine

**Development only.**  The data this provider serves is synthetic fixture data
in the *style* of Indian railway timetables.  It is not live railway data, it
is not an IRCTC API, and it must never be presented to users as current
schedules or availability.  A future *authorised* production provider will
replace this adapter behind the same Phase 2A models.

Responsibilities (deliberately narrow)
--------------------------------------
* load a small deterministic timetable fixture from JSON;
* map it into the existing Phase 2A contract models -- ``Station``, ``Train``,
  ``RailwayStationStop`` and ``TrainSchedule`` -- so every invariant is
  enforced by Phase 2A validation on construction (no validation is duplicated
  here);
* answer three deterministic timetable questions: station lookup, train
  lookup / schedule lookup, and trains between two stations.

Out of scope (by design)
------------------------
* no seat availability, fares, RAC/WL, quotas or any inventory claim: this
  provider implements timetable responsibility #1 only.  Availability stays
  with the existing :class:`engine.fixture_provider.FixtureAvailabilityProvider`
  protocol path, and the Phase 2A ``AvailabilitySnapshot`` is never produced
  here.
* no HTTP, no API keys, no caching policy, no database.

Not-found behaviour follows the repository's established provider pattern
(:class:`engine.fixture_provider.FixtureAvailabilityProvider`): a source with
nothing to say returns ``None`` (or an empty tuple for the route query) --
an unknown train or station is never silently invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from engine import time_utils as tu
from engine.errors import DataSourceError, DomainValidationError
from engine.fixtures import DEFAULT_FIXTURE_DIRECTORY
from engine.models import normalize_station_code
from engine.railway_data import RailwayStationStop, Station, Train, TrainSchedule

__all__ = [
    "DEFAULT_TIMETABLE_FIXTURE",
    "TrainsBetweenStationsResult",
    "TimetableFixtureProvider",
]

#: Repository-relative location of the shipped development timetable fixture.
DEFAULT_TIMETABLE_FIXTURE = (
    DEFAULT_FIXTURE_DIRECTORY / "timetable" / "timetable.json"
)

#: Provenance identifier: the name itself marks every derived record as
#: development fixture data, so fixture records can never masquerade as a
#: live or authorised source.
DEFAULT_PROVIDER_NAME = "development-timetable-fixture"


@dataclass(frozen=True, slots=True)
class TrainsBetweenStationsResult:
    """The deterministic answer to one trains-between-stations query.

    :param origin_station_code: normalized origin of the query.
    :param destination_station_code: normalized destination of the query.
    :param service_date: the service date the query was asked for, verbatim.
    :param trains: matching trains in deterministic order (origin departure,
        then train number).  Empty when no fixture train serves the pair in
        route order -- a reversed or unknown pair returns no trains and is
        never "fixed" by reversing a route.
    """

    origin_station_code: str
    destination_station_code: str
    service_date: date
    trains: tuple[Train, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "origin_station_code", normalize_station_code(self.origin_station_code)
        )
        object.__setattr__(
            self,
            "destination_station_code",
            normalize_station_code(self.destination_station_code),
        )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataSourceError(f"timetable fixture file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DataSourceError(f"timetable fixture {path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise DataSourceError(
            f"timetable fixture {path} must be a JSON object, got {type(document).__name__}"
        )
    return document


def _require_mapping_list(document: dict[str, Any], key: str, path: Path) -> list[Any]:
    value = document.get(key)
    if not isinstance(value, list):
        raise DataSourceError(f"timetable fixture {path} must contain a {key!r} JSON array")
    return value


def _stop_from_json(entry: dict[str, Any], path: Path) -> RailwayStationStop:
    """Map one fixture JSON stop into the Phase 2A stop (validated on build)."""
    try:
        return RailwayStationStop(
            station_code=entry["station_code"],
            sequence=entry["sequence"],
            arrival=tu.parse_clock(entry["arrival"]) if entry.get("arrival") else None,
            departure=tu.parse_clock(entry["departure"]) if entry.get("departure") else None,
            day_offset=entry.get("day_offset", 0),
            departure_day_offset=entry.get("departure_day_offset"),
        )
    except KeyError as exc:
        raise DataSourceError(
            f"timetable fixture {path} stop entry is missing required key {exc.args[0]!r}"
        ) from exc


class TimetableFixtureProvider:
    """Development timetable provider backed by the fixture JSON file.

    :param directory: a directory containing ``timetable/timetable.json``;
        defaults to the repository's shipped fixture snapshot.
    :param path: a direct path to a timetable fixture JSON file.  Mutually
        exclusive with ``directory``.
    :param provider_name: provenance identifier for this provider instance.

    Every record is built through the Phase 2A models, so invalid fixture data
    (bad station code, non-increasing sequences, non-chronological times,
    missing origin departure or terminus arrival) raises
    :class:`engine.errors.DomainValidationError` at load time -- the adapter
    adds no validation rules of its own.
    """

    def __init__(
        self,
        *,
        directory: Path | str | None = None,
        path: Path | str | None = None,
        provider_name: str = DEFAULT_PROVIDER_NAME,
    ) -> None:
        if directory is not None and path is not None:
            raise DomainValidationError(
                "pass either a fixture directory or a fixture file path, not both"
            )
        if path is not None:
            fixture_path = Path(path)
        else:
            base = Path(directory) if directory is not None else DEFAULT_FIXTURE_DIRECTORY
            fixture_path = base / "timetable" / "timetable.json"

        document = _load_json(fixture_path)
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            raise DataSourceError(f"timetable fixture {fixture_path} metadata must be an object")
        if metadata.get("data_type") != "DEVELOPMENT_FIXTURE_DATA" or metadata.get(
            "is_live_data"
        ):
            raise DataSourceError(
                f"timetable fixture {fixture_path} must be marked as development fixture "
                "data (metadata.data_type == 'DEVELOPMENT_FIXTURE_DATA', "
                "metadata.is_live_data == false)"
            )
        self._metadata: dict[str, Any] = metadata
        self._fixture_path = fixture_path

        if not isinstance(provider_name, str) or not provider_name.strip():
            raise DomainValidationError("provider_name must be a non-empty string")
        self._provider_name = provider_name.strip()

        self._stations: dict[str, Station] = {}
        for entry in _require_mapping_list(document, "stations", fixture_path):
            if not isinstance(entry, dict):
                raise DataSourceError(
                    f"timetable fixture {fixture_path} station entries must be objects"
                )
            try:
                station = Station(
                    code=entry["code"],
                    name=entry["name"],
                    city=entry.get("city"),
                    timezone=entry.get("timezone"),
                )
            except KeyError as exc:
                raise DataSourceError(
                    f"timetable fixture {fixture_path} station entry is missing required "
                    f"key {exc.args[0]!r}"
                ) from exc
            if station.code in self._stations:
                raise DataSourceError(
                    f"timetable fixture {fixture_path} defines station {station.code} twice"
                )
            self._stations[station.code] = station

        self._trains: dict[str, Train] = {}
        for entry in _require_mapping_list(document, "trains", fixture_path):
            if not isinstance(entry, dict):
                raise DataSourceError(
                    f"timetable fixture {fixture_path} train entries must be objects"
                )
            try:
                train = Train(
                    train_number=entry["train_number"],
                    train_name=entry["train_name"],
                    stops=tuple(_stop_from_json(stop, fixture_path) for stop in entry["stops"]),
                )
            except KeyError as exc:
                raise DataSourceError(
                    f"timetable fixture {fixture_path} train entry is missing required "
                    f"key {exc.args[0]!r}"
                ) from exc
            if train.train_number in self._trains:
                raise DataSourceError(
                    f"timetable fixture {fixture_path} defines train "
                    f"{train.train_number} twice"
                )
            self._trains[train.train_number] = train

    # ------------------------------------------------------------------
    # provenance
    # ------------------------------------------------------------------
    @property
    def provider_name(self) -> str:
        """Provenance identifier of this provider."""
        return self._provider_name

    @property
    def is_development_fixture(self) -> bool:
        """Always ``True``: this provider serves development fixture data only."""
        return True

    @property
    def fixture_metadata(self) -> dict[str, Any]:
        """The fixture file's own metadata block (read-only copy)."""
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # timetable lookups (responsibility #1 only -- never availability)
    # ------------------------------------------------------------------
    def get_station(self, station_code: str) -> Station | None:
        """The normalized station for ``station_code``, or ``None`` if unknown.

        The code is normalised with Phase 1's rules first, so ``"blr"`` finds
        ``BLR``.  An unknown station returns ``None``; it is never invented.
        """
        return self._stations.get(normalize_station_code(station_code))

    def get_train(self, train_number: str) -> Train | None:
        """The normalized train (with its ordered route), or ``None`` if unknown."""
        from engine.models import normalize_train_number

        return self._trains.get(normalize_train_number(train_number))

    def get_schedule(self, train_number: str, service_date: date) -> TrainSchedule | None:
        """The train's normalized schedule on ``service_date``, or ``None``.

        The schedule carries the ``service_date`` verbatim together with the
        train's explicit stop ``day_offset`` values; offsets are never inferred
        from the naive clock times.  An unknown train returns ``None`` -- no
        schedule is silently created for it.
        """
        train = self.get_train(train_number)
        if train is None:
            return None
        return TrainSchedule(train=train, service_date=service_date)

    def trains_between_stations(
        self,
        origin_station_code: str,
        destination_station_code: str,
        service_date: date,
    ) -> TrainsBetweenStationsResult:
        """Deterministically list fixture trains serving origin -> destination.

        Rules (mirroring the Phase 2A contract's route semantics):

        * the origin must occur strictly before the destination along a train's
          route -- routes are never reversed, and an equal pair is never served;
        * only trains serving both stations are returned, and no train is
          invented;
        * results are ordered deterministically by origin departure minute,
          then train number, so the same fixture always answers identically.

        Unknown stations simply yield an empty result -- consistent with the
        provider not-found pattern used across the repository.
        """
        origin = normalize_station_code(origin_station_code)
        destination = normalize_station_code(destination_station_code)
        matched = [
            train
            for train in self._trains.values()
            if train.serves_in_order(origin, destination)
        ]
        matched.sort(
            key=lambda train: (
                train.stop_for(origin).effective_departure_minutes,
                train.train_number,
            )
        )
        return TrainsBetweenStationsResult(
            origin_station_code=origin,
            destination_station_code=destination,
            service_date=service_date,
            trains=tuple(matched),
        )
