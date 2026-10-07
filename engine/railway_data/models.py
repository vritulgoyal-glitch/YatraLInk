"""Normalized railway data contract (Phase 2A).

The journey engine must never depend on a provider's response format.  Between
an authorised railway data source and the deterministic engine sits one small,
provider-neutral vocabulary::

    External railway provider
              |
      provider adapter            <-- Phase 2B: transport + field mapping + coercion
              |
      THIS CONTRACT                 <-- Phase 2A: the models below
              |
      Phase 1 journey engine
              |
      ranked journey alternatives

Design rules
------------
* **Reuse, never duplicate.**  Station codes, train numbers, availability states,
  travel classes, fares and clock arithmetic come from Phase 1
  (:mod:`engine.models`, :mod:`engine.enums`, :mod:`engine.money`,
  :mod:`engine.time_utils`).  ``TrainClass`` is literally Phase 1's
  :class:`engine.enums.TravelClass`; only :class:`Quota` is new here, because
  Phase 1 keeps a quota as a free non-empty string.
* **Money is integer paise.**  A float or bool fare is rejected, never rounded.
* **Wall clocks stay wall clocks, instants stay instants.**  Timetable times are
  naive, minute-granular wall-clock times and keep their ``day_offset``, so a
  stop after midnight is unambiguous.  ``fetched_at`` / ``updated_at`` are
  instants and must be timezone-aware.
* **Nothing is invented.**  A snapshot with no reported inventory is ``UNKNOWN``
  -- never ``AVAILABLE``, and never carrying an availability counter.
* **Provenance travels.**  Every external-data snapshot carries its provider name
  and fetch time, and an update time when the source supplies one.

The models are pure data: they validate their own invariants on construction and
expose the derived values the engine needs.  They perform no I/O, know no API
keys, open no sockets and parse no provider payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from enum import StrEnum, unique

from engine import time_utils as tu
from engine.enums import AvailabilityState, TravelClass, coerce_travel_class
from engine.errors import DomainValidationError, UnknownStationError
from engine.models import Station as EngineStation
from engine.models import Train as EngineTrain
from engine.models import TrainStop as EngineTrainStop
from engine.models import normalize_station_code, normalize_train_number
from engine.money import Fare
from engine.railway_data.validation import (
    as_tzinfo,
    localise,
    require_aware_datetime,
    require_non_empty_text,
    require_paise,
    require_schedule_date,
    require_timezone_name,
    require_unique_stations,
    schedule_minutes,
    validate_day_offsets,
    validate_sequence,
    validate_state_counters,
    validate_stop_times,
)

__all__ = [
    "AvailabilitySnapshot",
    "Quota",
    "RailwayStationStop",
    "Station",
    "Train",
    "TrainClass",
    "TrainSchedule",
    "coerce_quota",
    "normalize_train_class",
    "normalize_quota",
]

#: The normalized travel-class identifier *is* Phase 1's enum: one source of
#: truth, so a class can never mean two different things on either side of the
#: contract.
TrainClass = TravelClass

#: Phase 1's coercion, re-exported under the contract's name.
normalize_train_class = coerce_travel_class


@unique
class Quota(StrEnum):
    """Normalized reservation quota identifiers.

    Phase 1 keeps a quota as a free non-empty uppercase string, so no enum is
    duplicated here: this enum only *names* the quotas a provider adapter is
    expected to map into, and its values are exactly the strings the engine
    already accepts.  ``Quota.GENERAL.value == "GENERAL"``, which is Phase 1's
    :data:`engine.search.DEFAULT_QUOTA`.
    """

    GENERAL = "GENERAL"
    TATKAL = "TATKAL"
    PREMIUM_TATKAL = "PREMIUM_TATKAL"
    LADIES = "LADIES"
    SENIOR_CITIZEN = "SENIOR_CITIZEN"
    DIVYAANG = "DIVYAANG"


def coerce_quota(value: Quota | str) -> Quota:
    """Coerce ``value`` to :class:`Quota` or raise.

    Surrounding whitespace and case are normalised, so ``" tatkal "`` is
    ``Quota.TATKAL``.  Anything else is rejected -- a provider adapter must map
    its own quota code rather than pass it through.
    """
    if isinstance(value, Quota):
        return value
    if isinstance(value, str):
        candidate = value.strip().upper().replace(" ", "_").replace("-", "_")
        try:
            return Quota(candidate)
        except ValueError as exc:
            raise DomainValidationError(
                f"unsupported quota {value!r}; expected one of {[q.value for q in Quota]}"
            ) from exc
    raise DomainValidationError(f"quota must be a string, got {type(value).__name__}")


#: Readable alias for :func:`coerce_quota`.
normalize_quota = coerce_quota


@dataclass(frozen=True, slots=True)
class Station:
    """A railway station in normalized form.

    ``code`` is the identity used everywhere in the engine and is normalised by
    Phase 1's rules (uppercase, ``[A-Z][A-Z0-9]{1,9}``).  ``name`` is display
    text and is *not* assumed unique -- two stations may legitimately share a
    name.

    This is the contract's richer view of Phase 1's :class:`engine.models.Station`
    (it adds the two optional fields the contract requires), and
    :meth:`to_engine_station` bridges back to the engine's own type.
    """

    code: str
    name: str
    city: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", normalize_station_code(self.code))
        object.__setattr__(
            self, "name", require_non_empty_text(self.name, f"station {self.code} name")
        )
        if self.city is not None:
            object.__setattr__(
                self, "city", require_non_empty_text(self.city, f"station {self.code} city")
            )
        if self.timezone is not None:
            object.__setattr__(
                self,
                "timezone",
                require_timezone_name(self.timezone, f"station {self.code} timezone"),
            )

    @property
    def tzinfo(self) -> tzinfo | None:
        """The station's timezone as a :class:`zoneinfo.ZoneInfo`, or ``None``."""
        return None if self.timezone is None else as_tzinfo(self.timezone)

    def to_engine_station(self) -> EngineStation:
        """The Phase 1 station for this contract station (city/timezone dropped)."""
        return EngineStation(code=self.code, name=self.name)

    def to_json(self) -> dict[str, object]:
        return {
            "code": self.code,
            "name": self.name,
            "city": self.city,
            "timezone": self.timezone,
        }


@dataclass(frozen=True, slots=True)
class RailwayStationStop:
    """One scheduled stop of a train, in normalized (provider-neutral) form.

    :param station_code: canonical station code.
    :param sequence: 1-based position along the route; positive and strictly
        increasing across a route.
    :param arrival: naive, minute-granular wall-clock arrival time, or ``None``
        at the train's origin.
    :param departure: naive, minute-granular wall-clock departure time, or
        ``None`` at the train's terminus.
    :param day_offset: whole days after the service day on which ``arrival``
        occurs.  Preserved exactly as reported: it is what keeps an arrival after
        midnight unambiguous.
    :param departure_day_offset: ``day_offset``, or ``day_offset + 1`` for a halt
        that itself crosses midnight.  Same meaning as Phase 1's field of the
        same name, which is why the mapping in :meth:`to_engine_stop` is lossless.
    """

    station_code: str
    sequence: int
    arrival: time | None = None
    departure: time | None = None
    day_offset: int = 0
    departure_day_offset: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "station_code", normalize_station_code(self.station_code))
        tu.require_int(self.sequence, "RailwayStationStop.sequence", minimum=1)
        effective = validate_stop_times(
            station_code=self.station_code,
            sequence=self.sequence,
            arrival=self.arrival,
            departure=self.departure,
            day_offset=self.day_offset,
            departure_day_offset=self.departure_day_offset,
            label="stop",
        )
        object.__setattr__(self, "departure_day_offset", effective)

    @property
    def arrival_minutes(self) -> int | None:
        """Minutes from the service day's midnight to this stop's arrival."""
        if self.arrival is None:
            return None
        return schedule_minutes(self.day_offset, self.arrival, self.station_code)

    @property
    def departure_minutes(self) -> int | None:
        """Minutes from the service day's midnight to this stop's departure."""
        if self.departure is None:
            return None
        return schedule_minutes(self.departure_day_offset, self.departure, self.station_code)

    @property
    def effective_arrival_minutes(self) -> int:
        """Arrival minute for chronology, falling back to the departure."""
        value = self.arrival_minutes
        return value if value is not None else self.departure_minutes  # type: ignore[return-value]

    @property
    def effective_departure_minutes(self) -> int:
        """Departure minute for chronology, falling back to the arrival."""
        value = self.departure_minutes
        return value if value is not None else self.arrival_minutes  # type: ignore[return-value]

    def arrival_at(self, service_date: date, timezone: tzinfo | str) -> datetime | None:
        """Aware arrival instant, or ``None`` at the origin.

        The service date plus ``day_offset`` is *localised* to ``timezone`` --
        never converted -- so the published clock time and calendar day survive
        unchanged.
        """
        if self.arrival is None:
            return None
        day = require_schedule_date(service_date, "service_date") + timedelta(days=self.day_offset)
        return localise(day, self.arrival, timezone)

    def departure_at(self, service_date: date, timezone: tzinfo | str) -> datetime | None:
        """Aware departure instant, or ``None`` at the terminus."""
        if self.departure is None:
            return None
        day = require_schedule_date(service_date, "service_date") + timedelta(
            days=self.departure_day_offset
        )
        return localise(day, self.departure, timezone)

    def to_engine_stop(self) -> EngineTrainStop:
        """The equivalent Phase 1 stop."""
        return EngineTrainStop(
            station_code=self.station_code,
            sequence=self.sequence,
            arrival=self.arrival,
            departure=self.departure,
            day_offset=self.day_offset,
            departure_day_offset=self.departure_day_offset,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "station_code": self.station_code,
            "sequence": self.sequence,
            "arrival": tu.format_clock(self.arrival) if self.arrival else None,
            "departure": tu.format_clock(self.departure) if self.departure else None,
            "day_offset": self.day_offset,
            "departure_day_offset": self.departure_day_offset,
        }

    def __str__(self) -> str:  # pragma: no cover - diagnostics helper
        arrival = tu.format_clock(self.arrival) if self.arrival else "--:--"
        departure = tu.format_clock(self.departure) if self.departure else "--:--"
        return f"{self.sequence}:{self.station_code} {arrival}/{departure} (+{self.day_offset})"


@dataclass(frozen=True, slots=True)
class Train:
    """A train and its ordered route, in normalized form.

    The route is the single source of truth for direction: the train serves
    ``origin -> destination`` **only** when origin appears strictly before
    destination.  Routes are never reversed, and the exact reported stop order is
    preserved.
    """

    train_number: str
    train_name: str
    stops: tuple[RailwayStationStop, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_number", normalize_train_number(self.train_number))
        object.__setattr__(
            self,
            "train_name",
            require_non_empty_text(self.train_name, f"train {self.train_number} name"),
        )

        stops = tuple(self.stops)
        for stop in stops:
            if not isinstance(stop, RailwayStationStop):
                raise DomainValidationError(
                    f"train {self.train_number} stops must be RailwayStationStop, "
                    f"got {type(stop).__name__}"
                )
        if len(stops) < 2:
            raise DomainValidationError(
                f"train {self.train_number} must have at least 2 stops, got {len(stops)}"
            )
        object.__setattr__(self, "stops", stops)

        validate_sequence([stop.sequence for stop in stops], f"train {self.train_number}")
        require_unique_stations(
            [stop.station_code for stop in stops], f"train {self.train_number} route"
        )
        validate_day_offsets([stop.day_offset for stop in stops], f"train {self.train_number}")
        if stops[0].departure is None:
            raise DomainValidationError(
                f"train {self.train_number} origin stop must have a departure time"
            )
        if stops[-1].arrival is None:
            raise DomainValidationError(
                f"train {self.train_number} terminus stop must have an arrival time"
            )
        for position, stop in enumerate(stops):
            if position > 0 and stop.arrival is None:
                raise DomainValidationError(
                    f"train {self.train_number} stop {stop.sequence} ({stop.station_code}) "
                    "is not the origin but has no arrival time"
                )
            if position < len(stops) - 1 and stop.departure is None:
                raise DomainValidationError(
                    f"train {self.train_number} stop {stop.sequence} ({stop.station_code}) "
                    "is not the terminus but has no departure time"
                )

        previous: RailwayStationStop | None = None
        for stop in stops:
            if (
                previous is not None
                and stop.effective_arrival_minutes < previous.effective_departure_minutes
            ):
                raise DomainValidationError(
                    f"train {self.train_number} is not chronological: stop {stop.sequence} "
                    f"({stop.station_code}) occurs before stop {previous.sequence} "
                    f"({previous.station_code})"
                )
            previous = stop

    # ------------------------------------------------------------------
    # route lookups (mirroring Phase 1 so mapping is mechanical)
    # ------------------------------------------------------------------
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
        """True when this route contains ``station_code``."""
        code = normalize_station_code(station_code)
        return any(stop.station_code == code for stop in self.stops)

    def stop_for(self, station_code: str) -> RailwayStationStop:
        """The stop record for ``station_code``; raises when not served."""
        code = normalize_station_code(station_code)
        for stop in self.stops:
            if stop.station_code == code:
                return stop
        raise UnknownStationError(f"train {self.train_number} does not serve station {code}")

    def stop_index(self, station_code: str) -> int:
        """Zero-based position of ``station_code`` along the route."""
        code = normalize_station_code(station_code)
        for index, stop in enumerate(self.stops):
            if stop.station_code == code:
                return index
        raise UnknownStationError(f"train {self.train_number} does not serve station {code}")

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

    def stops_between(
        self, origin_code: str, destination_code: str
    ) -> tuple[RailwayStationStop, ...]:
        """Ordered stops from origin to destination, inclusive; ``()`` if invalid."""
        if not self.serves_in_order(origin_code, destination_code):
            return ()
        return self.stops[
            self.stop_index(origin_code) : self.stop_index(destination_code) + 1
        ]

    def to_engine_train(self) -> EngineTrain:
        """The equivalent Phase 1 train, valid by Phase 1's own rules."""
        return EngineTrain(
            number=self.train_number,
            name=self.train_name,
            stops=tuple(stop.to_engine_stop() for stop in self.stops),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "train_number": self.train_number,
            "train_name": self.train_name,
            "stops": [stop.to_json() for stop in self.stops],
        }


@dataclass(frozen=True, slots=True)
class TrainSchedule:
    """A train's routine plus the service date it is being run on.

    :param train: the normalized train and its ordered route.
    :param service_date: the calendar date on which the train's ``day_offset == 0``
        begins.  A stop with a higher offset simply lands on a later day; the date
        is never shifted to "fix" an overnight arrival.
    :param stops: the ordered stops, defaulting to the train's own route.  When
        supplied it must describe exactly the same ordered route, which lets an
        adapter be explicit without letting it contradict the train.
    """

    train: Train
    service_date: date
    stops: tuple[RailwayStationStop, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.train, Train):
            raise DomainValidationError(
                f"TrainSchedule.train must be a Train, got {type(self.train).__name__}"
            )
        object.__setattr__(
            self, "service_date", require_schedule_date(self.service_date, "service_date")
        )
        if self.stops is None:
            object.__setattr__(self, "stops", self.train.stops)
            return
        stops = tuple(self.stops)
        for stop in stops:
            if not isinstance(stop, RailwayStationStop):
                raise DomainValidationError(
                    f"schedule stops must be RailwayStationStop, got {type(stop).__name__}"
                )
        supplied = tuple((stop.sequence, stop.station_code) for stop in stops)
        expected = tuple((stop.sequence, stop.station_code) for stop in self.train.stops)
        if supplied != expected:
            raise DomainValidationError(
                f"schedule stops {supplied} do not match train {self.train.train_number} "
                f"route {expected}"
            )
        object.__setattr__(self, "stops", stops)

    @property
    def train_number(self) -> str:
        return self.train.train_number

    @property
    def origin_code(self) -> str:
        return self.train.origin_code

    @property
    def terminus_code(self) -> str:
        return self.train.terminus_code

    @property
    def resolved_stops(self) -> tuple[RailwayStationStop, ...]:
        """The ordered stops this schedule runs (never ``None``)."""
        return self.train.stops if self.stops is None else self.stops

    def stop_for(self, station_code: str) -> RailwayStationStop:
        """The scheduled stop for a station on this route."""
        return self.train.stop_for(station_code)

    def service_date_for(self, stop: RailwayStationStop) -> date:
        """Calendar date on which ``stop`` occurs: service date + its day offset."""
        return self.service_date + timedelta(days=stop.day_offset)

    def arrival_at(self, station_code: str, timezone: tzinfo | str) -> datetime | None:
        """Aware arrival instant at ``station_code``, or ``None`` at the origin."""
        return self.stop_for(station_code).arrival_at(self.service_date, timezone)

    def departure_at(self, station_code: str, timezone: tzinfo | str) -> datetime | None:
        """Aware departure instant from ``station_code``, or ``None`` at the terminus."""
        return self.stop_for(station_code).departure_at(self.service_date, timezone)

    def to_engine_train(self) -> EngineTrain:
        """The equivalent Phase 1 train (a schedule's date is not part of it)."""
        return self.train.to_engine_train()

    def to_json(self) -> dict[str, object]:
        return {
            "train_number": self.train_number,
            "service_date": self.service_date.isoformat(),
            "stops": [stop.to_json() for stop in self.resolved_stops],
        }


@dataclass(frozen=True, slots=True)
class AvailabilitySnapshot:
    """What a provider reported for one bookable unit, normalized.

    A bookable unit is one ``train / travel date / class / quota / origin-destination``
    combination.

    :param state: the reported inventory state, defaulting to
        :attr:`AvailabilityState.UNKNOWN`.  ``UNKNOWN`` means "the source did not
        report inventory" and is **never** confirmed availability.
    :param available_count: seats reported for ``AVAILABLE`` only.
    :param rac_count: RAC places reported for ``RAC`` only.
    :param waitlist_value: waitlist position reported for ``WAITLIST`` only.
    :param fare_paise: fare in integer paise, or ``None`` when unreported.  Never
        a float.
    :param provider: stable identifier of the data source.  Required.
    :param fetched_at: when the snapshot was retrieved; a timezone-aware instant.
        Required, and never invented by the engine.
    :param updated_at: when the source says the data was last updated, if it says.
    """

    train_number: str
    origin_station_code: str
    destination_station_code: str
    travel_date: date
    travel_class: TrainClass | str
    provider: str
    fetched_at: datetime
    state: AvailabilityState | str = AvailabilityState.UNKNOWN
    quota: Quota | str = Quota.GENERAL
    available_count: int | None = None
    rac_count: int | None = None
    waitlist_value: int | None = None
    fare_paise: int | None = None
    updated_at: datetime | None = None

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
            raise DomainValidationError(
                "availability snapshot origin and destination must differ, got "
                f"{self.origin_station_code} twice"
            )
        object.__setattr__(
            self, "travel_date", require_schedule_date(self.travel_date, "travel_date")
        )
        object.__setattr__(self, "travel_class", normalize_train_class(self.travel_class))
        object.__setattr__(self, "quota", coerce_quota(self.quota))
        object.__setattr__(
            self,
            "provider",
            require_non_empty_text(self.provider, "availability snapshot provider"),
        )
        object.__setattr__(
            self, "fetched_at", require_aware_datetime(self.fetched_at, "fetched_at")
        )
        if self.updated_at is not None:
            updated = require_aware_datetime(self.updated_at, "updated_at")
            if updated > self.fetched_at:
                raise DomainValidationError(
                    f"updated_at {updated.isoformat()} is later than fetched_at "
                    f"{self.fetched_at.isoformat()}; a snapshot cannot report an update "
                    "that happened after it was fetched"
                )
            object.__setattr__(self, "updated_at", updated)

        object.__setattr__(
            self,
            "state",
            validate_state_counters(
                self.state,
                available_count=self.available_count,
                rac_count=self.rac_count,
                waitlist_value=self.waitlist_value,
                label=f"snapshot for train {self.train_number}",
            ),
        )
        if self.fare_paise is not None:
            require_paise(self.fare_paise, "fare_paise")

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def unreported(
        cls,
        *,
        train_number: str,
        origin_station_code: str,
        destination_station_code: str,
        travel_date: date,
        travel_class: TrainClass | str,
        provider: str,
        fetched_at: datetime,
        quota: Quota | str = Quota.GENERAL,
        updated_at: datetime | None = None,
    ) -> AvailabilitySnapshot:
        """A snapshot for a unit the source had nothing to say about.

        The state is ``UNKNOWN`` and every counter and fare is absent: a missing
        report is recorded as a gap, never upgraded into availability.
        """
        return cls(
            train_number=train_number,
            origin_station_code=origin_station_code,
            destination_station_code=destination_station_code,
            travel_date=travel_date,
            travel_class=travel_class,
            provider=provider,
            fetched_at=fetched_at,
            state=AvailabilityState.UNKNOWN,
            quota=quota,
            updated_at=updated_at,
        )

    # ------------------------------------------------------------------
    # derived values
    # ------------------------------------------------------------------
    @property
    def key(self) -> tuple[object, ...]:
        """Identity of the bookable unit: everything that distinguishes it."""
        return (
            self.train_number,
            self.origin_station_code,
            self.destination_station_code,
            self.travel_date,
            self.travel_class.value,
            self.quota.value,
        )

    @property
    def is_confirmed(self) -> bool:
        """True only for a reported ``AVAILABLE`` state; ``UNKNOWN`` never is."""
        return self.state.is_confirmed

    @property
    def fare(self) -> Fare | None:
        """The fare as an exact Phase 1 :class:`engine.money.Fare`, or ``None``."""
        return None if self.fare_paise is None else Fare.from_paise(self.fare_paise)

    def to_availability_record(self) -> object:
        """The Phase 1 provider record for this snapshot.

        ``fetched_at`` / ``updated_at`` are converted to UTC and then stripped of
        their offset, because Phase 1's provider boundary transports naive
        datetimes: the *instant* is preserved exactly, only the labelling changes.
        ``rac_count`` / ``waitlist_value`` have no slot on that record -- the
        engine reads them through ``state`` -- and are deliberately not folded
        into ``seats_available``.
        """
        from engine.provider import AvailabilityRecord

        return AvailabilityRecord(
            train_number=self.train_number,
            origin_station_code=self.origin_station_code,
            destination_station_code=self.destination_station_code,
            travel_date=self.travel_date,
            travel_class=self.travel_class,
            state=self.state,
            quota=self.quota.value,
            seats_available=self.available_count,
            fare=self.fare,
            provider=self.provider,
            fetched_at=_as_utc_naive(self.fetched_at),
            updated_at=None if self.updated_at is None else _as_utc_naive(self.updated_at),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "train_number": self.train_number,
            "origin_station_code": self.origin_station_code,
            "destination_station_code": self.destination_station_code,
            "travel_date": self.travel_date.isoformat(),
            "travel_class": self.travel_class.value,
            "quota": self.quota.value,
            "state": self.state.value,
            "available_count": self.available_count,
            "rac_count": self.rac_count,
            "waitlist_value": self.waitlist_value,
            "fare_paise": self.fare_paise,
            "provider": self.provider,
            "fetched_at": self.fetched_at.isoformat(),
            "updated_at": None if self.updated_at is None else self.updated_at.isoformat(),
        }


def _as_utc_naive(value: datetime) -> datetime:
    """Convert an aware instant to the naive UTC form Phase 1 transports."""
    return value.astimezone(UTC).replace(tzinfo=None)
