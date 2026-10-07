"""The availability-provider boundary.

The journey engine must never know *where* availability comes from (JSON
fixtures, a database, a live railway API, a third party).  This module defines
the small contract that decouples the two sides:

* :class:`AvailabilityProvider` — what a data source implements.  Its job is
  **data access only**: given an :class:`AvailabilityQuery`, return an
  :class:`AvailabilityRecord` (or ``None`` when the source has nothing to say).
* :class:`AvailabilityLookup` — what the engine consumes. 
  :class:`engine.availability.AvailabilityBook` already satisfies it
  structurally, so the existing in-memory path is untouched.
* :class:`ProviderAvailabilityBook` — the adapter that lets the pipeline accept
  any :class:`AvailabilityProvider` and present it as an
  :class:`AvailabilityLookup`.

The provider never generates journeys, validates connections or ranks anything;
the engine never parses JSON, opens sockets or knows a provider's name unless
the record carries it.  A future live railway provider implements
:class:`AvailabilityProvider` and the deterministic algorithms stay exactly as
they are.

Freshness metadata
------------------
Every :class:`AvailabilityRecord` can carry *when* the data was fetched
(``fetched_at``) and, when the upstream source reports it, *when it was last
updated* (``updated_at``).  Phase 1 only transports these timestamps: no
staleness policy is applied and stale data is **never** automatically
downgraded — that is a deliberate future policy decision.

``UNKNOWN`` semantics
---------------------
A provider signals "the source did not report this bookable unit" by returning
``None``.  It may also report an explicit ``UNKNOWN`` state.  Both reach the
engine as :attr:`AvailabilityState.UNKNOWN`, which is a distinct non-confirmed
state — only :attr:`AvailabilityState.AVAILABLE` is ever confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from engine import time_utils as tu
from engine.availability import Availability, DateAgnosticAvailability
from engine.enums import (
    AvailabilityState,
    TravelClass,
    coerce_availability_state,
    coerce_travel_class,
)
from engine.errors import DomainValidationError
from engine.money import Fare

__all__ = [
    "AvailabilityLookup",
    "AvailabilityProvider",
    "AvailabilityQuery",
    "AvailabilityRecord",
    "ProviderAvailabilityBook",
    "availability_from_record",
]


@dataclass(frozen=True, slots=True)
class AvailabilityQuery:
    """One bookable unit a provider may be asked about.

    A bookable unit is one ``train / travel date / class / quota / from-to``
    combination — the same identity an :class:`engine.availability.Availability`
    carries, expressed as a question.
    """

    train_number: str
    origin_station_code: str
    destination_station_code: str
    travel_date: date
    travel_class: TravelClass | str
    quota: str = "GENERAL"

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
                "availability query origin and destination must differ, got "
                f"{self.origin_station_code} twice"
            )
        tu.require_date(self.travel_date, "AvailabilityQuery.travel_date")
        object.__setattr__(self, "travel_class", coerce_travel_class(self.travel_class))
        if not isinstance(self.quota, str) or not self.quota.strip():
            raise DomainValidationError("AvailabilityQuery.quota must be a non-empty string")
        object.__setattr__(self, "quota", self.quota.strip().upper())

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


@dataclass(frozen=True, slots=True)
class AvailabilityRecord:
    """What a provider reports for one bookable unit.

    ``state`` is authoritative and must be one of the five reported states;
    ``seats_available`` and ``fare`` are optional extra facts.  ``provider``,
    ``fetched_at`` and ``updated_at`` are provenance/freshness metadata: they
    are transported verbatim and no engine behaviour depends on them (yet).
    """

    train_number: str
    origin_station_code: str
    destination_station_code: str
    travel_date: date
    travel_class: TravelClass | str
    state: AvailabilityState | str
    quota: str = "GENERAL"
    seats_available: int | None = None
    fare: Fare | None = None
    provider: str = ""
    fetched_at: datetime | None = None
    updated_at: datetime | None = None

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
                "availability record origin and destination must differ, got "
                f"{self.origin_station_code} twice"
            )
        tu.require_date(self.travel_date, "AvailabilityRecord.travel_date")
        object.__setattr__(self, "travel_class", coerce_travel_class(self.travel_class))
        object.__setattr__(self, "state", coerce_availability_state(self.state))
        if not isinstance(self.quota, str) or not self.quota.strip():
            raise DomainValidationError("AvailabilityRecord.quota must be a non-empty string")
        object.__setattr__(self, "quota", self.quota.strip().upper())
        if self.seats_available is not None:
            tu.require_int(self.seats_available, "AvailabilityRecord.seats_available", minimum=0)
        if self.fare is not None and not isinstance(self.fare, Fare):
            raise DomainValidationError("AvailabilityRecord.fare must be a Fare or None")
        if not isinstance(self.provider, str):
            raise DomainValidationError("AvailabilityRecord.provider must be a string")
        if self.fetched_at is not None:
            tu.require_datetime(self.fetched_at, "AvailabilityRecord.fetched_at")
        if self.updated_at is not None:
            tu.require_datetime(self.updated_at, "AvailabilityRecord.updated_at")

    @property
    def is_confirmed(self) -> bool:
        """True only when the reported state is confirmed availability.

        ``UNKNOWN`` — whether reported explicitly or signalled by a provider
        returning ``None`` — is *never* confirmed.
        """
        state = coerce_availability_state(self.state)
        return state.is_confirmed


@runtime_checkable
class AvailabilityProvider(Protocol):
    """The contract every availability data source implements.

    Implementations own **data access only**: reading JSON, querying a
    database, calling a live railway API — and nothing else.  Journey
    generation, connection validation and ranking stay in the engine.

    Returning ``None`` means "the source did not report this bookable unit";
    the engine renders that as :attr:`AvailabilityState.UNKNOWN`.  A provider
    may equally return a record whose state *is* ``UNKNOWN``.
    """

    @property
    def provider_name(self) -> str:
        """Stable identifier of the data source, for provenance reporting."""
        ...

    def get_availability(self, query: AvailabilityQuery) -> AvailabilityRecord | None:
        """Reported inventory for one bookable unit, or ``None`` if unreported."""
        ...


@runtime_checkable
class AvailabilityLookup(Protocol):
    """The engine-facing availability view.

    :class:`engine.availability.AvailabilityBook` satisfies this structurally,
    which is why the engine can consume either the in-memory book or a
    provider-backed adapter without a single algorithmic change.
    """

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
        """The reported record for one bookable unit, or ``None`` if unreported."""
        ...


def availability_from_record(record: AvailabilityRecord) -> Availability:
    """Convert a provider record into the engine's domain availability.

    ``updated_at`` is preferred for ``source_updated_at``; when the upstream
    source does not report an update time, ``fetched_at`` is used so the
    freshness provenance still travels with the domain object.
    """
    if not isinstance(record, AvailabilityRecord):
        raise DomainValidationError(
            f"expected an AvailabilityRecord, got {type(record).__name__}"
        )
    source_updated_at = record.updated_at if record.updated_at is not None else record.fetched_at
    return Availability(
        train_number=record.train_number,
        origin_station_code=record.origin_station_code,
        destination_station_code=record.destination_station_code,
        travel_date=record.travel_date,
        travel_class=record.travel_class,
        state=record.state,
        quota=record.quota,
        seats_available=record.seats_available,
        fare=record.fare,
        source_updated_at=source_updated_at,
    )


class ProviderAvailabilityBook:
    """Adapter: presents any :class:`AvailabilityProvider` as a lookup source.

    The pipeline hands this object to the assembly step exactly where it would
    hand an :class:`engine.availability.AvailabilityBook`; the assembly step
    cannot tell the difference and does not need to.  Results are cached per
    query key, so a provider is asked once per bookable unit per run.
    """

    __slots__ = ("_provider", "_cache")

    def __init__(self, provider: AvailabilityProvider) -> None:
        if not callable(getattr(provider, "get_availability", None)):
            raise DomainValidationError(
                "provider must implement get_availability(query) -> AvailabilityRecord | None"
            )
        self._provider = provider
        self._cache: dict[tuple[object, ...], Availability | None] = {}

    @property
    def provider_name(self) -> str:
        """The wrapped provider's identifier."""
        name = getattr(self._provider, "provider_name", "")
        return name if isinstance(name, str) else ""

    def lookup(
        self,
        *,
        train_number: str,
        origin_station_code: str,
        destination_station_code: str,
        travel_date: date,
        travel_class: TravelClass | str,
        quota: str = "GENERAL",
    ) -> Availability | None:
        """Ask the provider, cache the answer, return the engine-domain view."""
        query = AvailabilityQuery(
            train_number=train_number,
            origin_station_code=origin_station_code,
            destination_station_code=destination_station_code,
            travel_date=travel_date,
            travel_class=travel_class,
            quota=quota,
        )
        if query.key not in self._cache:
            record = self._provider.get_availability(query)
            self._cache[query.key] = (
                availability_from_record(record) if record is not None else None
            )
        return self._cache[query.key]
