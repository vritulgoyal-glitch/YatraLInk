"""Rail legs: the primitive "travel from stop *i* to stop *j* on one train" unit.

A :class:`RailLeg` is a *schedulable* piece of travel: it knows which train, which
two stops of that train's route, and which calendar occurrence of the train is
being used (its anchor date).  Everything else — fare, availability, tickets,
risk — belongs to the journey layer, not here.

Legs are always generated in **route order** and never reversed: a leg exists
only when the boarding stop precedes the alighting stop in the train's own route.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from engine import time_utils as tu
from engine.config import SearchConfiguration
from engine.errors import DomainValidationError
from engine.models import Train, normalize_station_code
from engine.network import Timetable

__all__ = ["RailLeg", "boarding_anchor", "legs_from_station"]


@dataclass(frozen=True, slots=True)
class RailLeg:
    """One uninterrupted ride on one train between two of its stops."""

    train: Train
    origin_index: int
    destination_index: int
    anchor_date: date

    def __post_init__(self) -> None:
        if not isinstance(self.train, Train):
            raise DomainValidationError(
                f"RailLeg.train must be a Train, got {type(self.train).__name__}"
            )
        count = len(self.train.stops)
        tu.require_int(self.origin_index, "RailLeg.origin_index", minimum=0)
        tu.require_int(self.destination_index, "RailLeg.destination_index", minimum=0)
        if self.origin_index >= count or self.destination_index >= count:
            raise DomainValidationError(
                f"RailLeg indices out of range for train {self.train.number} "
                f"({count} stops): {self.origin_index} -> {self.destination_index}"
            )
        if self.origin_index >= self.destination_index:
            raise DomainValidationError(
                f"RailLeg must travel forwards along the route of train "
                f"{self.train.number}: {self.origin_index} -> {self.destination_index}"
            )
        if self.boarding_stop.departure is None:
            raise DomainValidationError(
                f"train {self.train.number} cannot be boarded at "
                f"{self.boarding_stop.station_code}: the stop has no departure time"
            )
        if self.alighting_stop.arrival is None:
            raise DomainValidationError(
                f"train {self.train.number} cannot be left at "
                f"{self.alighting_stop.station_code}: the stop has no arrival time"
            )

    # ------------------------------------------------------------------
    # route facts
    # ------------------------------------------------------------------
    @property
    def boarding_stop(self):  # type: ignore[no-untyped-def]
        """The stop where the ride begins."""
        return self.train.stops[self.origin_index]

    @property
    def alighting_stop(self):  # type: ignore[no-untyped-def]
        """The stop where the ride ends."""
        return self.train.stops[self.destination_index]

    @property
    def origin_station_code(self) -> str:
        return self.boarding_stop.station_code

    @property
    def destination_station_code(self) -> str:
        return self.alighting_stop.station_code

    @property
    def train_number(self) -> str:
        return self.train.number

    @property
    def stop_count(self) -> int:
        """Number of scheduled stops between boarding and alighting, exclusive."""
        return self.destination_index - self.origin_index - 1

    # ------------------------------------------------------------------
    # absolute times
    # ------------------------------------------------------------------
    @property
    def departure(self) -> datetime:
        """Absolute departure datetime of this ride."""
        stop = self.boarding_stop
        return tu.stop_datetime(
            self.anchor_date,
            stop.departure_day_offset,
            stop.departure,  # type: ignore[arg-type]
        )

    @property
    def arrival(self) -> datetime:
        """Absolute arrival datetime of this ride."""
        stop = self.alighting_stop
        return tu.stop_datetime(self.anchor_date, stop.day_offset, stop.arrival)  # type: ignore[arg-type]

    @property
    def duration_minutes(self) -> int:
        """Elapsed minutes on board (always positive)."""
        return tu.duration_minutes(self.departure, self.arrival, context=f"leg {self.train_number}")

    @property
    def boarding_sequence(self) -> int:
        return self.boarding_stop.sequence

    @property
    def alighting_sequence(self) -> int:
        return self.alighting_stop.sequence

    def describe(self) -> str:
        """Short readable form, used in rejection reports and diagnostics."""
        return (
            f"{self.train_number} {self.origin_station_code}"
            f"->{self.destination_station_code} "
            f"({self.departure.isoformat()} -> {self.arrival.isoformat()})"
        )

    def __str__(self) -> str:  # pragma: no cover - diagnostics helper
        return self.describe()


def boarding_anchor(train: Train, boarding_index: int, threshold: datetime) -> date:
    """Anchor date of the earliest occurrence boarding at or after ``threshold``."""
    stop = train.stops[boarding_index]
    if stop.departure is None:
        raise DomainValidationError(
            f"train {train.number} stop {stop.sequence} ({stop.station_code}) has no "
            "departure time and cannot be boarded"
        )
    return tu.boarding_anchor_date(threshold, stop.departure, stop.departure_day_offset)  # type: ignore[arg-type]


def legs_from_station(
    timetable: Timetable,
    station_code: str,
    threshold: datetime,
    *,
    config: SearchConfiguration,
    deadline: datetime | None = None,
    excluded_trains: frozenset[str] = frozenset(),
    excluded_stations: frozenset[str] = frozenset(),
    destination_filter: str | None = None,
) -> tuple[RailLeg, ...]:
    """Every forward leg that can be boarded at ``station_code`` at/after ``threshold``.

    For each train serving the station, exactly one occurrence is considered: the
    **earliest** daily occurrence whose departure is not before ``threshold``.
    This is a deliberate Phase 1 simplification that keeps the candidate set
    finite and deterministic — the engine never invents extra runs of a service.

    Results are ordered by train number and then by position along the route, so
    the output is fully reproducible.

    :param deadline: drop legs that would arrive after this moment.
    :param excluded_trains: train numbers that must not be used.
    :param excluded_stations: stations the leg may not travel to.
    :param destination_filter: when given, only legs ending at that station.
    """
    station = normalize_station_code(station_code)
    target = normalize_station_code(destination_filter) if destination_filter else None
    legs: list[RailLeg] = []

    for train in timetable.trains_at(station):
        if train.number in excluded_trains:
            continue
        boarding_index = train.stop_index(station)
        if train.stops[boarding_index].departure is None:
            # The station is the train's terminus: nothing can be boarded here.
            continue
        anchor = boarding_anchor(train, boarding_index, threshold)
        for index in range(boarding_index + 1, len(train.stops)):
            stop = train.stops[index]
            if stop.station_code in excluded_stations:
                continue
            if target is not None and stop.station_code != target:
                continue
            leg = RailLeg(
                train=train,
                origin_index=boarding_index,
                destination_index=index,
                anchor_date=anchor,
            )
            if deadline is not None and leg.arrival > deadline:
                continue
            legs.append(leg)

    return tuple(legs)
