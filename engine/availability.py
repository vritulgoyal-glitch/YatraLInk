"""Availability representation and deterministic aggregation.

Availability is *reported* by a data source, never inferred.  The engine stores
exactly what it was told and treats "not told" as :attr:`AvailabilityState.UNKNOWN`
— which is a distinct, non-bookable state, never a synonym for AVAILABLE.

Aggregation rule for a multi-segment journey
--------------------------------------------
Each segment carries one state.  The journey's aggregate state is the **worst**
state on the desirability ladder::

    AVAILABLE (0)  <  RAC (1)  <  WAITLIST (2)  <  UNKNOWN (3)  <  NOT_AVAILABLE (4)

so the aggregate is the maximum-desirability-index state across segments.  This
matters in two documented ways:

* ``UNKNOWN`` outranks ``NOT_AVAILABLE`` (nothing has been reported as
  impossible), but it ranks *below* ``WAITLIST`` and is **never** treated as
  confirmed.  A journey is confirmed only when every segment is ``AVAILABLE``.
* A journey with one ``NOT_AVAILABLE`` segment aggregates to ``NOT_AVAILABLE``
  and is rejected by the pipeline before ranking.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime

from engine import time_utils as tu
from engine.enums import (
    AVAILABILITY_DESIRABILITY,
    AVAILABILITY_ORDER,
    AvailabilityState,
    TravelClass,
    coerce_availability_state,
    coerce_travel_class,
)
from engine.errors import DomainValidationError
from engine.money import Fare

__all__ = [
    "Availability",
    "AvailabilityBook",
    "AvailabilitySummary",
    "DateAgnosticAvailability",
    "summarize_availability",
]


@dataclass(frozen=True, slots=True)
class Availability:
    """Reported inventory for one bookable unit.

    A bookable unit is one ``train / travel date / class / quota / from-to``
    combination.  ``state`` is authoritative; ``seats_available`` and ``fare``
    are optional extra facts and may be absent without changing the state.
    """

    train_number: str
    origin_station_code: str
    destination_station_code: str
    travel_date: date
    travel_class: TravelClass
    state: AvailabilityState
    quota: str = "GENERAL"
    seats_available: int | None = None
    fare: Fare | None = None
    source_updated_at: datetime | None = None

    def __post_init__(self) -> None:
        from engine.models import normalize_station_code, normalize_train_number

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
            raise DomainValidationError(
                "availability origin and destination must differ, got "
                f"{self.origin_station_code} twice"
            )
        tu.require_date(self.travel_date, "Availability.travel_date")
        object.__setattr__(self, "travel_class", coerce_travel_class(self.travel_class))
        object.__setattr__(self, "state", coerce_availability_state(self.state))
        if not isinstance(self.quota, str) or not self.quota.strip():
            raise DomainValidationError("Availability.quota must be a non-empty string")
        object.__setattr__(self, "quota", self.quota.strip().upper())
        if self.seats_available is not None:
            tu.require_int(self.seats_available, "Availability.seats_available", minimum=0)
        if self.fare is not None and not isinstance(self.fare, Fare):
            raise DomainValidationError("Availability.fare must be a Fare or None")
        if self.source_updated_at is not None:
            tu.require_datetime(self.source_updated_at, "Availability.source_updated_at")

    @property
    def key(self) -> tuple[object, ...]:
        """Lookup key: everything that identifies the bookable unit."""
        return (
            self.train_number,
            self.origin_station_code,
            self.destination_station_code,
            self.travel_date,
            self.travel_class.value,
            self.quota,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "train_number": self.train_number,
            "origin_station_code": self.origin_station_code,
            "destination_station_code": self.destination_station_code,
            "travel_date": self.travel_date.isoformat(),
            "travel_class": self.travel_class.value,
            "quota": self.quota,
            "state": self.state.value,
            "seats_available": self.seats_available,
            "fare": self.fare.to_json() if self.fare else None,
        }


@dataclass(frozen=True, slots=True)
class AvailabilitySummary:
    """Aggregate availability of a journey, with per-state counts."""

    state: AvailabilityState
    counts: Mapping[AvailabilityState, int]
    segment_count: int

    @property
    def all_confirmed(self) -> bool:
        """True when every segment is ``AVAILABLE`` and there is at least one."""
        return (
            self.segment_count > 0
            and self.counts.get(AvailabilityState.AVAILABLE, 0) == self.segment_count
        )

    @property
    def has_unknown(self) -> bool:
        return self.counts.get(AvailabilityState.UNKNOWN, 0) > 0

    @property
    def has_not_available(self) -> bool:
        return self.counts.get(AvailabilityState.NOT_AVAILABLE, 0) > 0

    @property
    def worst_segment_states(self) -> tuple[AvailabilityState, ...]:
        return tuple(state for state, count in self.counts.items() if count and state is self.state)

    def describe(self) -> str:
        """Short human-readable explanation of the aggregate state."""
        parts = [
            f"{count}x{state.value}"
            for state in AVAILABILITY_ORDER
            if (count := self.counts.get(state, 0))
        ]
        return f"{self.state.value} ({', '.join(parts)})" if parts else self.state.value

    def to_json(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "segment_count": self.segment_count,
            "all_confirmed": self.all_confirmed,
            "counts": {
                state.value: self.counts.get(state, 0)
                for state in AVAILABILITY_ORDER
                if state in self.counts
            },
        }


def summarize_availability(states: Iterable[AvailabilityState]) -> AvailabilitySummary:
    """Aggregate segment states into a journey-level summary.

    Deterministic and order-independent: the result depends only on the multiset
    of states, never on the order the segments are visited.
    """
    materialised = tuple(coerce_availability_state(state) for state in states)
    counts: dict[AvailabilityState, int] = {}
    for state in materialised:
        counts[state] = counts.get(state, 0) + 1
    if not materialised:
        return AvailabilitySummary(state=AvailabilityState.UNKNOWN, counts={}, segment_count=0)
    worst = max(counts, key=lambda state: AVAILABILITY_DESIRABILITY[state])
    ordered_counts = {state: counts[state] for state in AVAILABILITY_ORDER if state in counts}
    return AvailabilitySummary(state=worst, counts=ordered_counts, segment_count=len(materialised))


@dataclass(frozen=True, slots=True)
class DateAgnosticAvailability:
    """Availability that the snapshot declares independent of the travel date.

    Modelled as its own type rather than as an ``Availability`` with a missing
    date so that the two can never be confused: a dated record proves *when* it
    was reported, while this record only proves *what* was reported.
    """

    train_number: str
    origin_station_code: str
    destination_station_code: str
    travel_class: TravelClass
    state: AvailabilityState
    quota: str = "GENERAL"
    seats_available: int | None = None
    fare: Fare | None = None

    def __post_init__(self) -> None:
        from engine.models import normalize_station_code, normalize_train_number

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
            raise DomainValidationError(
                "availability origin and destination must differ, got "
                f"{self.origin_station_code} twice"
            )
        object.__setattr__(self, "travel_class", coerce_travel_class(self.travel_class))
        object.__setattr__(self, "state", coerce_availability_state(self.state))
        if not isinstance(self.quota, str) or not self.quota.strip():
            raise DomainValidationError("Availability.quota must be a non-empty string")
        object.__setattr__(self, "quota", self.quota.strip().upper())
        if self.seats_available is not None:
            tu.require_int(self.seats_available, "Availability.seats_available", minimum=0)
        if self.fare is not None and not isinstance(self.fare, Fare):
            raise DomainValidationError("Availability.fare must be a Fare or None")

    @property
    def key(self) -> tuple[object, ...]:
        """Lookup key without the travel date."""
        return (
            self.train_number,
            self.origin_station_code,
            self.destination_station_code,
            self.travel_class.value,
            self.quota,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "train_number": self.train_number,
            "origin_station_code": self.origin_station_code,
            "destination_station_code": self.destination_station_code,
            "travel_date": None,
            "travel_class": self.travel_class.value,
            "quota": self.quota,
            "state": self.state.value,
            "seats_available": self.seats_available,
            "fare": self.fare.to_json() if self.fare else None,
        }


class AvailabilityBook:
    """Immutable lookup of availability by bookable-unit key.

    The book is a plain in-memory snapshot.  Phase 1 populates it from synthetic
    fixtures; a future railway-data adapter can populate exactly the same object
    without any change to the engine.

    An entry may also be *date-independent* (``travel_date=None``).  That is an
    explicit affordance for synthetic snapshots, where one declaration stands in
    for "this service reports the same state on every date in the snapshot".  A
    dated entry always wins over a date-independent one, and the engine still
    returns :attr:`AvailabilityState.UNKNOWN` when neither exists — it never
    invents a state.
    """

    __slots__ = ("_entries", "_by_lookup", "_by_lookup_any_date")

    def __init__(self, entries: Iterable[Availability | DateAgnosticAvailability] = ()) -> None:
        materialised = tuple(entries)
        self._entries = materialised
        index: dict[tuple[object, ...], Availability] = {}
        any_date_index: dict[tuple[object, ...], Availability] = {}
        for entry in materialised:
            if isinstance(entry, DateAgnosticAvailability):
                key = entry.key
                if key in any_date_index:
                    raise DomainValidationError(
                        "duplicate date-independent availability entry for "
                        f"{entry.train_number} {entry.origin_station_code}->"
                        f"{entry.destination_station_code} {entry.travel_class.value}/{entry.quota}"
                    )
                any_date_index[key] = entry
                continue
            if not isinstance(entry, Availability):
                raise DomainValidationError(
                    "AvailabilityBook entries must be Availability or "
                    f"DateAgnosticAvailability, got {type(entry).__name__}"
                )
            if entry.key in index:
                raise DomainValidationError(
                    "duplicate availability entry for "
                    f"{entry.train_number} {entry.origin_station_code}->"
                    f"{entry.destination_station_code} {entry.travel_date} "
                    f"{entry.travel_class.value}/{entry.quota}"
                )
            index[entry.key] = entry
        self._by_lookup = index
        self._by_lookup_any_date = any_date_index

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[Availability | DateAgnosticAvailability, ...]:
        """All entries, in their original order."""
        return self._entries

    def lookup(
        self,
        *,
        train_number: str,
        origin_station_code: str,
        destination_station_code: str,
        travel_date: date,
        travel_class: TravelClass | str,
        quota: str = "GENERAL",
    ) -> Availability | DateAgnosticAvailability | None:
        """Fetch the availability record for one bookable unit.

        Returns ``None`` when the data source did not report this unit.  The
        caller must translate that into :attr:`AvailabilityState.UNKNOWN`; the
        book never invents a state.  A dated entry takes precedence over a
        date-independent one.
        """
        from engine.models import normalize_station_code, normalize_train_number

        klass = coerce_travel_class(travel_class)
        number = normalize_train_number(train_number)
        origin = normalize_station_code(origin_station_code)
        destination = normalize_station_code(destination_station_code)
        day = tu.require_date(travel_date, "travel_date")
        normalised_quota = quota.strip().upper()
        dated = self._by_lookup.get(
            (number, origin, destination, day, klass.value, normalised_quota)
        )
        if dated is not None:
            return dated
        return self._by_lookup_any_date.get(
            (number, origin, destination, klass.value, normalised_quota)
        )
