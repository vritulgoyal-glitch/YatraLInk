"""Domain models for the YatraLink railway engine.

The models are deliberately *pure data*: they validate their own invariants on
construction and expose derived values, but the algorithms that combine them
live in dedicated modules (:mod:`engine.connections`, :mod:`engine.journey`,
:mod:`engine.candidates`, :mod:`engine.ranking`).

Two representation decisions matter for correctness:

* **Stop times** are ``(clock time, day offset)`` pairs, so a route that crosses
  midnight is represented exactly and never compared as a bare clock time.
* **Money** is an integer number of paise (:class:`engine.money.Fare`), never a
  float.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time

from engine import time_utils as tu
from engine.availability import AvailabilitySummary, summarize_availability
from engine.enums import (
    AvailabilityState,
    ConnectionKind,
    ConnectionRisk,
    JourneyType,
    coerce_availability_state,
)
from engine.errors import DomainValidationError, InvalidJourneyError, UnknownStationError
from engine.money import Fare, sum_fares

__all__ = [
    "ConnectionInfo",
    "JourneyOption",
    "JourneySegment",
    "Station",
    "Train",
    "TrainStop",
    "normalize_station_code",
    "normalize_train_number",
]

#: Station codes are short, uppercase, alphanumeric identifiers (e.g. ``BLR``).
STATION_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")

#: A train number is an uppercase alphanumeric identifier (e.g. ``YT1001``).
TRAIN_NUMBER_RE = re.compile(r"^[A-Z0-9][A-Z0-9-]{1,15}$")

#: Fixed anchor date used only to compare two clock times of the *same stop*.
_COMPARISON_ANCHOR = date(2000, 1, 1)


def normalize_station_code(raw: str) -> str:
    """Normalise a station code to its canonical form.

    Station codes are compared and stored uppercase, so ``"blr"`` and ``" BLR "``
    are the same station.  Anything that cannot be a station code raises.
    """
    if not isinstance(raw, str):
        raise DomainValidationError(f"station code must be a string, got {type(raw).__name__}")
    code = raw.strip().upper()
    if not code:
        raise DomainValidationError("station code must not be empty")
    if not STATION_CODE_RE.match(code):
        raise DomainValidationError(
            f"invalid station code {raw!r}: expected 2-10 characters, "
            "starting with a letter, using A-Z and 0-9"
        )
    return code


def normalize_train_number(raw: str) -> str:
    """Normalise a train number to its canonical form."""
    if not isinstance(raw, str):
        raise DomainValidationError(f"train number must be a string, got {type(raw).__name__}")
    number = raw.strip().upper()
    if not number:
        raise DomainValidationError("train number must not be empty")
    if not TRAIN_NUMBER_RE.match(number):
        raise DomainValidationError(
            f"invalid train number {raw!r}: expected 2-16 characters using A-Z, 0-9 and '-'"
        )
    return number


@dataclass(frozen=True, slots=True)
class Station:
    """A railway station.

    ``code`` is the canonical identity used everywhere in the engine.  ``name``
    is display text and is *not* assumed to be globally unique: two different
    stations may legitimately share a name.
    """

    code: str
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", normalize_station_code(self.code))
        if not isinstance(self.name, str) or not self.name.strip():
            raise DomainValidationError(f"station {self.code} must have a non-empty name")
        object.__setattr__(self, "name", self.name.strip())

    def to_json(self) -> dict[str, str]:
        return {"code": self.code, "name": self.name}


@dataclass(frozen=True, slots=True)
class TrainStop:
    """One scheduled stop of a train.

    :param station_code: canonical station code.
    :param sequence: 1-based order of the stop within the train's route.
    :param arrival: clock time of arrival, or ``None`` at the train's origin.
    :param departure: clock time of departure, or ``None`` at the train's terminus.
    :param day_offset: days after the route anchor date on which ``arrival`` occurs.
    :param departure_day_offset: days after the anchor for ``departure``; defaults
        to ``day_offset`` and may only be ``day_offset + 1`` more, which is how a
        halt that itself crosses midnight is represented.
    """

    station_code: str
    sequence: int
    arrival: time | None = None
    departure: time | None = None
    day_offset: int = 0
    departure_day_offset: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "station_code", normalize_station_code(self.station_code))
        tu.require_int(self.sequence, "TrainStop.sequence", minimum=1)
        tu.require_int(self.day_offset, "TrainStop.day_offset", minimum=0)
        if self.arrival is None and self.departure is None:
            raise DomainValidationError(
                f"stop {self.sequence} at {self.station_code} must have an arrival "
                "time, a departure time, or both"
            )
        if self.arrival is not None:
            tu.require_clock(self.arrival, "TrainStop.arrival")
        if self.departure is not None:
            tu.require_clock(self.departure, "TrainStop.departure")

        if self.departure_day_offset is None:
            object.__setattr__(self, "departure_day_offset", self.day_offset)
        else:
            tu.require_int(self.departure_day_offset, "TrainStop.departure_day_offset", minimum=0)
            if self.departure_day_offset not in (self.day_offset, self.day_offset + 1):
                raise DomainValidationError(
                    f"stop {self.sequence} at {self.station_code}: "
                    "departure_day_offset must equal day_offset or day_offset + 1"
                )

        if self.arrival is not None and self.departure is not None:
            arrival_at = tu.stop_datetime(_COMPARISON_ANCHOR, self.day_offset, self.arrival)
            departure_at = tu.stop_datetime(
                _COMPARISON_ANCHOR, self.departure_day_offset, self.departure
            )
            if departure_at < arrival_at:
                raise DomainValidationError(
                    f"stop {self.sequence} at {self.station_code}: departure "
                    f"{tu.format_clock(self.departure)} (day +{self.departure_day_offset}) "
                    f"precedes arrival {tu.format_clock(self.arrival)} (day +{self.day_offset})"
                )

    # ------------------------------------------------------------------
    # derived values
    # ------------------------------------------------------------------
    @property
    def arrival_datetime(self) -> datetime | None:
        """Arrival as an anchor-relative datetime (``None`` at the origin stop)."""
        if self.arrival is None:
            return None
        return tu.stop_datetime(_COMPARISON_ANCHOR, self.day_offset, self.arrival)

    @property
    def departure_datetime(self) -> datetime | None:
        """Departure as an anchor-relative datetime (``None`` at the terminus)."""
        if self.departure is None:
            return None
        return tu.stop_datetime(_COMPARISON_ANCHOR, self.departure_day_offset, self.departure)

    @property
    def effective_arrival(self) -> datetime:
        """Arrival for chronology comparisons, falling back to departure."""
        value = self.arrival_datetime
        return value if value is not None else self.departure_datetime  # type: ignore[return-value]

    @property
    def effective_departure(self) -> datetime:
        """Departure for chronology comparisons, falling back to arrival."""
        value = self.departure_datetime
        return value if value is not None else self.arrival_datetime  # type: ignore[return-value]

    @property
    def halt_minutes(self) -> int | None:
        """Minutes spent at this stop, or ``None`` when the stop is one-sided."""
        if self.arrival is None or self.departure is None:
            return None
        return tu.minutes_between(self.arrival_datetime, self.departure_datetime)  # type: ignore[arg-type]

    def __str__(self) -> str:  # pragma: no cover - diagnostics helper
        arrival = tu.format_clock(self.arrival) if self.arrival else "--:--"
        departure = tu.format_clock(self.departure) if self.departure else "--:--"
        return f"{self.sequence}:{self.station_code} {arrival}/{departure} (+{self.day_offset})"


@dataclass(frozen=True, slots=True, eq=False)
class Train:
    """A train together with its ordered route.

    The route is the single source of truth for direction: a train only serves
    ``origin -> destination`` when ``origin`` appears **before** ``destination``
    in ``stops``.  Routes are never reversed by the engine.
    """

    number: str
    name: str
    stops: tuple[TrainStop, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "number", normalize_train_number(self.number))
        if not isinstance(self.name, str) or not self.name.strip():
            raise DomainValidationError(f"train {self.number} must have a non-empty name")
        object.__setattr__(self, "name", self.name.strip())

        stops = tuple(self.stops)
        if len(stops) < 2:
            raise DomainValidationError(
                f"train {self.number} must have at least 2 stops, got {len(stops)}"
            )
        object.__setattr__(self, "stops", stops)

        sequences = [stop.sequence for stop in stops]
        if sequences != sorted(set(sequences)):
            raise DomainValidationError(
                f"train {self.number} stop sequences must be strictly increasing, got {sequences}"
            )
        codes = [stop.station_code for stop in stops]
        if len(set(codes)) != len(codes):
            raise DomainValidationError(
                f"train {self.number} visits a station more than once: {codes}"
            )
        if stops[0].departure is None:
            raise DomainValidationError(f"train {self.number} origin stop must have a departure")
        if stops[-1].arrival is None:
            raise DomainValidationError(f"train {self.number} terminus stop must have an arrival")
        for position, stop in enumerate(stops):
            if position > 0 and stop.arrival is None:
                raise DomainValidationError(
                    f"train {self.number} stop {stop.sequence} ({stop.station_code}) "
                    "is not the origin but has no arrival time"
                )
            if position < len(stops) - 1 and stop.departure is None:
                raise DomainValidationError(
                    f"train {self.number} stop {stop.sequence} ({stop.station_code}) "
                    "is not the terminus but has no departure time"
                )

        previous: TrainStop | None = None
        for stop in stops:
            if previous is not None and stop.effective_arrival < previous.effective_departure:
                raise DomainValidationError(
                    f"train {self.number} is not chronological: stop {stop.sequence} "
                    f"({stop.station_code}) occurs before stop {previous.sequence} "
                    f"({previous.station_code})"
                )
            previous = stop

    # ------------------------------------------------------------------
    # route lookups
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Train):
            return NotImplemented
        return self.number == other.number and self.stops == other.stops

    def __hash__(self) -> int:
        return hash((self.number, self.stops))

    @property
    def origin_code(self) -> str:
        return self.stops[0].station_code

    @property
    def terminus_code(self) -> str:
        return self.stops[-1].station_code

    @property
    def station_codes(self) -> tuple[str, ...]:
        return tuple(stop.station_code for stop in self.stops)

    def serves(self, station_code: str) -> bool:
        """True when this train's route contains ``station_code``."""
        code = normalize_station_code(station_code)
        return any(stop.station_code == code for stop in self.stops)

    def stop_for(self, station_code: str) -> TrainStop:
        """The stop record for ``station_code``; raises when not served."""
        code = normalize_station_code(station_code)
        for stop in self.stops:
            if stop.station_code == code:
                return stop
        raise UnknownStationError(f"train {self.number} does not serve station {code}")

    def stop_index(self, station_code: str) -> int:
        """Zero-based position of ``station_code`` within the route."""
        code = normalize_station_code(station_code)
        for index, stop in enumerate(self.stops):
            if stop.station_code == code:
                return index
        raise UnknownStationError(f"train {self.number} does not serve station {code}")

    def serves_in_order(self, origin_code: str, destination_code: str) -> bool:
        """True when the route visits origin strictly before destination."""
        origin = normalize_station_code(origin_code)
        destination = normalize_station_code(destination_code)
        if origin == destination:
            return False
        try:
            return self.stop_index(origin) < self.stop_index(destination)
        except UnknownStationError:
            return False

    def stops_between(self, origin_code: str, destination_code: str) -> tuple[TrainStop, ...]:
        """Ordered stops from origin to destination, inclusive; ``()`` if invalid."""
        if not self.serves_in_order(origin_code, destination_code):
            return ()
        start = self.stop_index(origin_code)
        end = self.stop_index(destination_code)
        return self.stops[start : end + 1]

    def to_json(self) -> dict[str, object]:
        return {
            "number": self.number,
            "name": self.name,
            "stops": [
                {
                    "station_code": stop.station_code,
                    "sequence": stop.sequence,
                    "arrival": tu.format_clock(stop.arrival) if stop.arrival else None,
                    "departure": tu.format_clock(stop.departure) if stop.departure else None,
                    "day_offset": stop.day_offset,
                    "departure_day_offset": stop.departure_day_offset,
                }
                for stop in self.stops
            ],
        }


@dataclass(frozen=True, slots=True)
class JourneySegment:
    """One reservation-sized leg of a journey on one train.

    For a same-train split there are several segments with the **same**
    ``train_number``; ``reservation_index`` records the order of the separate
    tickets the passenger must hold.
    """

    train_number: str
    train_name: str
    origin_station_code: str
    destination_station_code: str
    departure: datetime
    arrival: datetime
    duration_minutes: int
    availability: AvailabilityState
    fare: Fare | None = None
    seats_available: int | None = None
    origin_station_name: str = ""
    destination_station_name: str = ""
    reservation_index: int = 0
    boarding_stop_sequence: int = 0
    alighting_stop_sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_number", normalize_train_number(self.train_number))
        object.__setattr__(
            self, "origin_station_code", normalize_station_code(self.origin_station_code)
        )
        object.__setattr__(
            self,
            "destination_station_code",
            normalize_station_code(self.destination_station_code),
        )
        if self.origin_station_code == self.destination_station_code:
            raise InvalidJourneyError(
                f"segment on train {self.train_number} has identical origin and "
                f"destination ({self.origin_station_code})"
            )
        tu.require_datetime(self.departure, "JourneySegment.departure")
        tu.require_datetime(self.arrival, "JourneySegment.arrival")
        tu.require_int(self.duration_minutes, "JourneySegment.duration_minutes", minimum=1)
        if self.arrival <= self.departure:
            raise InvalidJourneyError(
                f"segment on train {self.train_number} arrives at or before it departs "
                f"({self.departure.isoformat()} -> {self.arrival.isoformat()})"
            )
        actual = tu.minutes_between(self.departure, self.arrival)
        if actual != self.duration_minutes:
            raise InvalidJourneyError(
                f"segment on train {self.train_number} declares {self.duration_minutes} "
                f"minutes but spans {actual} minutes"
            )
        object.__setattr__(self, "availability", coerce_availability_state(self.availability))
        if self.fare is not None and not isinstance(self.fare, Fare):
            raise DomainValidationError("JourneySegment.fare must be a Fare or None")
        if self.seats_available is not None:
            tu.require_int(self.seats_available, "JourneySegment.seats_available", minimum=0)
        tu.require_int(self.reservation_index, "JourneySegment.reservation_index", minimum=0)

    @property
    def intermediate_stop_count(self) -> int:
        """Number of scheduled stops strictly between boarding and alighting."""
        if not self.boarding_stop_sequence or not self.alighting_stop_sequence:
            return 0
        return max(0, self.alighting_stop_sequence - self.boarding_stop_sequence - 1)

    @property
    def key(self) -> tuple[object, ...]:
        """Stable identity of this segment, used for deduplication."""
        return (
            self.train_number,
            self.origin_station_code,
            self.destination_station_code,
            self.departure,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "train_number": self.train_number,
            "train_name": self.train_name,
            "origin_station_code": self.origin_station_code,
            "origin_station_name": self.origin_station_name,
            "destination_station_code": self.destination_station_code,
            "destination_station_name": self.destination_station_name,
            "departure": self.departure.isoformat(),
            "arrival": self.arrival.isoformat(),
            "duration_minutes": self.duration_minutes,
            "availability": self.availability.value,
            "seats_available": self.seats_available,
            "fare": self.fare.to_json() if self.fare else None,
            "reservation_index": self.reservation_index,
        }


@dataclass(frozen=True, slots=True)
class ConnectionInfo:
    """The deterministic verdict on one transfer between adjacent segments."""

    index: int
    kind: ConnectionKind
    arrival: datetime
    departure: datetime
    transfer_minutes: int
    required_minutes: int
    valid: bool
    risk: ConnectionRisk
    from_station_code: str
    to_station_code: str
    buffer_minutes: int = 0
    note: str = ""

    @property
    def is_same_station(self) -> bool:
        return self.from_station_code == self.to_station_code

    @property
    def station_change(self) -> bool:
        return not self.is_same_station

    def to_json(self) -> dict[str, object]:
        return {
            "index": self.index,
            "kind": self.kind.value,
            "from_station_code": self.from_station_code,
            "to_station_code": self.to_station_code,
            "arrival": self.arrival.isoformat(),
            "departure": self.departure.isoformat(),
            "transfer_minutes": self.transfer_minutes,
            "required_minutes": self.required_minutes,
            "buffer_minutes": self.buffer_minutes,
            "valid": self.valid,
            "risk": self.risk.value,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class JourneyOption:
    """A complete, validated candidate journey from origin to destination.

    Every derived value (duration, fare, changes, risk, availability summary) is
    computed deterministically by :func:`engine.journey.build_journey_option`;
    this dataclass stores the result and guards its invariants.
    """

    journey_id: str
    journey_type: JourneyType
    origin_station_code: str
    destination_station_code: str
    segments: tuple[JourneySegment, ...]
    connections: tuple[ConnectionInfo, ...] = ()
    availability_summary: AvailabilitySummary | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "origin_station_code", normalize_station_code(self.origin_station_code)
        )
        object.__setattr__(
            self,
            "destination_station_code",
            normalize_station_code(self.destination_station_code),
        )
        segments = tuple(self.segments)
        if not segments:
            raise InvalidJourneyError(f"journey {self.journey_id} has no segments")
        if len(segments) > 3:
            raise InvalidJourneyError(
                f"journey {self.journey_id} has {len(segments)} segments; Phase 1 allows 3"
            )
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "connections", tuple(self.connections))
        if self.availability_summary is None:
            object.__setattr__(
                self,
                "availability_summary",
                summarize_availability(s.availability for s in segments),
            )
        if self.journey_type is JourneyType.DIRECT and len(segments) != 1:
            raise InvalidJourneyError(
                f"DIRECT journey {self.journey_id} must have exactly one segment"
            )
        if self.journey_type is JourneyType.SAME_TRAIN_SPLIT:
            if len(segments) < 2:
                raise InvalidJourneyError(
                    f"SAME_TRAIN_SPLIT journey {self.journey_id} needs at least two segments"
                )
            if len({s.train_number for s in segments}) != 1:
                raise InvalidJourneyError(
                    f"SAME_TRAIN_SPLIT journey {self.journey_id} spans multiple trains"
                )
        if self.journey_type is JourneyType.CONNECTING and len(segments) < 2:
            raise InvalidJourneyError(
                f"CONNECTING journey {self.journey_id} needs at least two segments"
            )

    # ------------------------------------------------------------------
    # derived structure
    # ------------------------------------------------------------------
    @property
    def departure(self) -> datetime:
        """First departure of the journey."""
        return self.segments[0].departure

    @property
    def arrival(self) -> datetime:
        """Final arrival of the journey."""
        return self.segments[-1].arrival

    @property
    def total_duration_minutes(self) -> int:
        """Elapsed time from first departure to final arrival.

        Elapsed time — **not** the sum of segment durations: transfer waiting is
        part of the journey and must be counted exactly once.
        """
        return tu.minutes_between(self.departure, self.arrival)

    @property
    def total_travel_minutes(self) -> int:
        """Time actually spent on board trains."""
        return sum(segment.duration_minutes for segment in self.segments)

    @property
    def total_transfer_minutes(self) -> int:
        """Time spent between segments, from one arrival to the next departure.

        Derived from the segments themselves rather than from the attached
        :class:`ConnectionInfo` records, so the identity
        ``total_duration == total_travel + total_transfer`` always holds.  The
        connection records remain the audit trail explaining *why* each transfer
        is or is not acceptable.
        """
        return sum(
            tu.minutes_between(previous.arrival, following.departure)
            for previous, following in zip(self.segments, self.segments[1:], strict=False)
        )

    @property
    def total_fare(self) -> Fare | None:
        """Sum of segment fares, or ``None`` when any segment fare is unknown."""
        return sum_fares(segment.fare for segment in self.segments)

    @property
    def train_changes(self) -> int:
        """Number of physical train transitions.

        A same-train split has zero changes even though it needs two tickets.
        """
        changes = 0
        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if previous.train_number != current.train_number:
                changes += 1
        return changes

    @property
    def reservation_count(self) -> int:
        """Number of separate tickets/reservations required."""
        return len(self.segments)

    @property
    def requires_separate_reservations(self) -> bool:
        """True when the journey cannot be covered by a single reservation."""
        return len(self.segments) > 1

    @property
    def availability(self) -> AvailabilityState:
        """Aggregate availability state of the weakest segment."""
        assert self.availability_summary is not None
        return self.availability_summary.state

    @property
    def risk(self) -> ConnectionRisk:
        """Worst connection risk across the journey (SAFE when direct)."""
        return max(
            (connection.risk for connection in self.connections),
            key=lambda risk: risk.desirability,
            default=ConnectionRisk.SAFE,
        )

    @property
    def is_confirmed(self) -> bool:
        """True only when *every* required segment is reported AVAILABLE."""
        assert self.availability_summary is not None
        return self.availability_summary.all_confirmed

    @property
    def train_numbers(self) -> tuple[str, ...]:
        return tuple(segment.train_number for segment in self.segments)

    @property
    def station_path(self) -> tuple[str, ...]:
        """Ordered station codes visited, including transfer points."""
        path = [self.segments[0].origin_station_code]
        for segment in self.segments:
            path.append(segment.destination_station_code)
        return tuple(path)

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------
    @property
    def dedup_key(self) -> tuple[object, ...]:
        """Canonical identity used for candidate deduplication.

        Includes the journey *type* so that a same-train split and a connecting
        journey with identical stations never collapse into one another.
        """
        return (
            self.journey_type.value,
            tuple(
                (
                    segment.train_number,
                    segment.origin_station_code,
                    segment.destination_station_code,
                )
                for segment in self.segments
            ),
            self.departure,
            self.arrival,
        )

    @property
    def stable_sort_key(self) -> tuple[object, ...]:
        """Final tiebreaker guaranteeing a total, stable ordering.

        The trailing journey id makes the order total: two candidates with the
        same structure resolve to the same key, and anything else resolves
        deterministically regardless of the order the caller supplied them in.
        """
        return (
            self.journey_type.value,
            self.train_numbers,
            self.station_path,
            self.departure,
            self.arrival,
            self.journey_id,
        )

    def __str__(self) -> str:  # pragma: no cover - diagnostics helper
        legs = " | ".join(
            f"{s.train_number} {s.origin_station_code}->{s.destination_station_code}"
            for s in self.segments
        )
        return f"{self.journey_type.value} [{legs}]"
