"""In-memory railway network catalog: stations, trains and transfer allowances.

:class:`Timetable` is the only structure the algorithms need in order to reason
about the railway.  It is built from plain data (see :mod:`engine.fixtures`) and
contains no I/O, no caching layer and no network access, so the engine can be
driven from a fixture file, a database or a future provider adapter without any
change to the algorithms.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from engine.errors import DataSourceError, DomainValidationError, UnknownStationError
from engine.models import Station, Train, normalize_station_code, normalize_train_number
from engine.time_utils import require_int

__all__ = ["TransferAllowance", "Timetable"]


@dataclass(frozen=True, slots=True)
class TransferAllowance:
    """Data-source statement that two *different* stations may be used to change trains.

    The engine never assumes that two distinct station codes are the same place.
    A station change is only ever considered when the supplied railway data
    contains an explicit allowance for that ordered station pair.

    Allowances are **directional**: ``DEL -> ANVT`` does not imply
    ``ANVT -> DEL``.  A provider that needs both directions declares both.
    """

    from_station_code: str
    to_station_code: str
    minimum_minutes: int
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "from_station_code", normalize_station_code(self.from_station_code)
        )
        object.__setattr__(self, "to_station_code", normalize_station_code(self.to_station_code))
        if self.from_station_code == self.to_station_code:
            raise DomainValidationError(
                "a transfer allowance must join two different stations, got "
                f"{self.from_station_code} twice"
            )
        require_int(self.minimum_minutes, "TransferAllowance.minimum_minutes", minimum=0)

    @property
    def key(self) -> tuple[str, str]:
        return (self.from_station_code, self.to_station_code)

    def to_json(self) -> dict[str, object]:
        return {
            "from_station_code": self.from_station_code,
            "to_station_code": self.to_station_code,
            "minimum_minutes": self.minimum_minutes,
            "note": self.note,
        }


class Timetable:
    """An immutable snapshot of stations and train routes.

    Construction validates the whole network and builds deterministic indexes,
    so that *any* iteration over the timetable is reproducible: trains at a
    station are always returned ordered by train number.
    """

    __slots__ = (
        "_stations",
        "_trains",
        "_station_by_code",
        "_trains_by_station",
        "_transfer_allowances",
    )

    def __init__(
        self,
        stations: Iterable[Station],
        trains: Iterable[Train],
        transfer_allowances: Iterable[TransferAllowance] = (),
    ) -> None:
        station_list = tuple(stations)
        train_list = tuple(trains)

        station_by_code: dict[str, Station] = {}
        for station in station_list:
            if not isinstance(station, Station):
                raise DomainValidationError(
                    f"timetable stations must be Station, got {type(station).__name__}"
                )
            if station.code in station_by_code:
                raise DomainValidationError(f"duplicate station code {station.code}")
            station_by_code[station.code] = station

        train_by_number: dict[str, Train] = {}
        for train in train_list:
            if not isinstance(train, Train):
                raise DomainValidationError(
                    f"timetable trains must be Train, got {type(train).__name__}"
                )
            if train.number in train_by_number:
                raise DomainValidationError(f"duplicate train number {train.number}")
            train_by_number[train.number] = train
            for code in train.station_codes:
                if code not in station_by_code:
                    raise DataSourceError(
                        f"train {train.number} stops at unknown station {code}; "
                        "the timetable must declare every station it uses"
                    )

        trains_by_station: dict[str, list[Train]] = {}
        for number in sorted(train_by_number):
            train = train_by_number[number]
            for code in train.station_codes:
                trains_by_station.setdefault(code, []).append(train)

        self._stations = tuple(sorted(station_list, key=lambda s: s.code))
        self._trains = tuple(train_by_number[number] for number in sorted(train_by_number))
        self._station_by_code = station_by_code
        self._trains_by_station = {
            code: tuple(trains) for code, trains in sorted(trains_by_station.items())
        }

        allowance_tuple = tuple(transfer_allowances)
        for allowance in allowance_tuple:
            if not isinstance(allowance, TransferAllowance):
                raise DomainValidationError(
                    "timetable transfer allowances must be TransferAllowance, got "
                    f"{type(allowance).__name__}"
                )
        #: Declared by the railway snapshot itself, so a caller can search a
        #: network without having to pass the allowances alongside it.
        self._transfer_allowances = allowance_tuple

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    @property
    def stations(self) -> tuple[Station, ...]:
        return self._stations

    @property
    def transfer_allowances(self) -> tuple[TransferAllowance, ...]:
        """Different-station transfer allowances declared by this snapshot.

        Empty when the data declares none — the engine then never allows a
        change between two different station codes.
        """
        return self._transfer_allowances

    @property
    def trains(self) -> tuple[Train, ...]:
        """All trains, ordered by train number (deterministic)."""
        return self._trains

    @property
    def station_codes(self) -> tuple[str, ...]:
        return tuple(station.code for station in self._stations)

    def has_station(self, station_code: str) -> bool:
        """True when ``station_code`` exists in this timetable."""
        return normalize_station_code(station_code) in self._station_by_code

    def station(self, station_code: str) -> Station:
        """Look up a station, raising :class:`UnknownStationError` if absent."""
        code = normalize_station_code(station_code)
        try:
            return self._station_by_code[code]
        except KeyError as exc:
            raise UnknownStationError(f"unknown station code {code}") from exc

    def station_name(self, station_code: str) -> str:
        """Display name of a station, or its code when it is not in the timetable."""
        try:
            return self.station(station_code).name
        except UnknownStationError:
            return normalize_station_code(station_code)

    def train(self, train_number: str) -> Train:
        """Look up a train by number, raising :class:`UnknownStationError` if absent."""
        number = normalize_train_number(train_number)
        for train in self._trains:
            if train.number == number:
                return train
        raise UnknownStationError(f"unknown train number {number}")

    def trains_at(self, station_code: str) -> tuple[Train, ...]:
        """Trains serving ``station_code``, ordered by train number."""
        code = normalize_station_code(station_code)
        return self._trains_by_station.get(code, ())

    def trains_between(self, origin_code: str, destination_code: str) -> tuple[Train, ...]:
        """Trains that serve ``origin_code`` strictly before ``destination_code``."""
        origin = normalize_station_code(origin_code)
        destination = normalize_station_code(destination_code)
        return tuple(train for train in self._trains if train.serves_in_order(origin, destination))

    def relevant_trains(self, origin_code: str, destination_code: str) -> tuple[Train, ...]:
        """Trains that could take part in a journey from origin to destination.

        A train is relevant when its route contains the origin, the destination,
        or both.  Trains serving neither endpoint can never contribute a segment
        to a journey between them.
        """
        origin = normalize_station_code(origin_code)
        destination = normalize_station_code(destination_code)
        return tuple(
            train for train in self._trains if train.serves(origin) or train.serves(destination)
        )

    def to_json(self) -> dict[str, object]:
        return {
            "stations": [station.to_json() for station in self._stations],
            "trains": [train.to_json() for train in self._trains],
        }


def lookup_transfer_allowance(
    from_station_code: str,
    to_station_code: str,
    allowances: Iterable[TransferAllowance],
) -> TransferAllowance | None:
    """Find the allowance for the ordered station pair, if the data declares one."""
    origin = normalize_station_code(from_station_code)
    destination = normalize_station_code(to_station_code)
    for allowance in allowances:
        if allowance.key == (origin, destination):
            return allowance
    return None
