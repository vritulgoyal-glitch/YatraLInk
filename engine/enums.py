"""Enumerations shared across the YatraLink deterministic journey engine.

Every enum in this module is a plain ``str``-valued enum so that values survive
JSON round-trips and are stable identifiers for API contracts.

Availability is intentionally modelled as a *ladder of desirability* rather than
as a flat set, because multi-segment journeys need a deterministic way to
aggregate several segment states into one journey state.  See
:mod:`engine.availability` for the aggregation policy.
"""

from __future__ import annotations

from enum import StrEnum, unique

__all__ = [
    "AVAILABILITY_DESIRABILITY",
    "AVAILABILITY_ORDER",
    "AvailabilityState",
    "CONNECTION_RISK_DESIRABILITY",
    "ConnectionKind",
    "ConnectionRisk",
    "JourneyType",
    "RejectionReason",
    "TravelClass",
    "coerce_availability_state",
    "coerce_travel_class",
]


@unique
class AvailabilityState(StrEnum):
    """Inventory state reported by a data source for one booking unit.

    A booking unit is one train / class / quota / date / origin-destination
    combination.  ``UNKNOWN`` means "the data source did not tell us"; it is
    never equivalent to confirmed availability.
    """

    AVAILABLE = "AVAILABLE"
    RAC = "RAC"
    WAITLIST = "WAITLIST"
    UNKNOWN = "UNKNOWN"
    NOT_AVAILABLE = "NOT_AVAILABLE"

    @property
    def desirability(self) -> int:
        """Rank on the desirability ladder (0 = best, 4 = worst)."""
        return AVAILABILITY_DESIRABILITY[self]

    @property
    def is_confirmed(self) -> bool:
        """True only for a reported ``AVAILABLE`` state."""
        return self is AvailabilityState.AVAILABLE

    @property
    def is_known(self) -> bool:
        """False for ``UNKNOWN`` (no information was reported)."""
        return self is not AvailabilityState.UNKNOWN

    @property
    def is_bookable(self) -> bool:
        """True for states a traveller can act on (available, RAC, waitlist)."""
        return self in _BOOKABLE_STATES


#: Desirability ladder: 0 is best.  ``UNKNOWN`` sits below ``WAITLIST`` because
#: unknown inventory is not actionable, and above ``NOT_AVAILABLE`` because
#: nothing has been reported as impossible.
AVAILABILITY_DESIRABILITY: dict[AvailabilityState, int] = {
    AvailabilityState.AVAILABLE: 0,
    AvailabilityState.RAC: 1,
    AvailabilityState.WAITLIST: 2,
    AvailabilityState.UNKNOWN: 3,
    AvailabilityState.NOT_AVAILABLE: 4,
}

#: Deterministic iteration order (best -> worst) for reports and diagnostics.
AVAILABILITY_ORDER: tuple[AvailabilityState, ...] = (
    AvailabilityState.AVAILABLE,
    AvailabilityState.RAC,
    AvailabilityState.WAITLIST,
    AvailabilityState.UNKNOWN,
    AvailabilityState.NOT_AVAILABLE,
)

_BOOKABLE_STATES = frozenset(
    {
        AvailabilityState.AVAILABLE,
        AvailabilityState.RAC,
        AvailabilityState.WAITLIST,
    }
)


@unique
class JourneyType(StrEnum):
    """How a journey is put together."""

    #: One reservation on one train for the whole origin -> destination leg.
    DIRECT = "DIRECT"
    #: Two or more reservations on the *same physical train* (e.g. BLR-HYD +
    #: HYD-DEL).  The passenger never changes trains, but the inventory and the
    #: fare of each reservation are evaluated independently.
    SAME_TRAIN_SPLIT = "SAME_TRAIN_SPLIT"
    #: At least one physical train change.
    CONNECTING = "CONNECTING"


@unique
class ConnectionKind(StrEnum):
    """Nature of a transfer between two consecutive journey segments."""

    #: Same train, separate reservation: the passenger stays on the train.
    SAME_TRAIN = "SAME_TRAIN"
    #: Different train, same station.
    CROSS_TRAIN_SAME_STATION = "CROSS_TRAIN_SAME_STATION"
    #: Different train *and* different station, allowed only when the fixture
    #: data explicitly provides a transfer allowance for that station pair.
    CROSS_STATION_TRANSFER = "CROSS_STATION_TRANSFER"


@unique
class ConnectionRisk(StrEnum):
    """Deterministic risk classification of a transfer."""

    SAFE = "SAFE"
    TIGHT = "TIGHT"
    INVALID = "INVALID"

    @property
    def desirability(self) -> int:
        """Rank on the desirability ladder (0 = best, 2 = worst)."""
        return CONNECTION_RISK_DESIRABILITY[self]


#: Desirability ladder for transfer risk: 0 is best.
CONNECTION_RISK_DESIRABILITY: dict[ConnectionRisk, int] = {
    ConnectionRisk.SAFE: 0,
    ConnectionRisk.TIGHT: 1,
    ConnectionRisk.INVALID: 2,
}


@unique
class RejectionReason(StrEnum):
    """Why a generated candidate was not returned as a journey option.

    Rejections are recorded rather than silently dropped, so that every
    candidate the generator produced can be accounted for.
    """

    #: A transfer between two segments cannot be made (missing buffer, station
    #: change without an allowance, or a departure that precedes the arrival).
    INVALID_CONNECTION = "INVALID_CONNECTION"
    #: Segments are not contiguous in time or station order.
    BROKEN_CHRONOLOGY = "BROKEN_CHRONOLOGY"
    #: A required segment is reported NOT_AVAILABLE, so the journey cannot be made.
    SEGMENT_NOT_AVAILABLE = "SEGMENT_NOT_AVAILABLE"
    #: Departure is before the traveller's earliest acceptable departure.
    BEFORE_EARLIEST_DEPARTURE = "BEFORE_EARLIEST_DEPARTURE"
    #: Arrival is after the traveller's latest acceptable arrival.
    AFTER_LATEST_ARRIVAL = "AFTER_LATEST_ARRIVAL"
    #: More train changes than the request or engine allows.
    TOO_MANY_TRAIN_CHANGES = "TOO_MANY_TRAIN_CHANGES"
    #: More segments than Phase 1 supports.
    TOO_MANY_SEGMENTS = "TOO_MANY_SEGMENTS"
    #: Structurally malformed candidate (e.g. non-contiguous stations).
    MALFORMED = "MALFORMED"
    #: Same candidate already produced by another generator.
    DUPLICATE = "DUPLICATE"


@unique
class TravelClass(StrEnum):
    """Reservation classes supported in Phase 1.

    The member names are readable identifiers; the values are the codes used by
    railway data sources and API contracts.
    """

    SLEEPER = "SL"
    AC_3_TIER = "3A"
    AC_2_TIER = "2A"
    AC_FIRST = "1A"
    CHAIR_CAR = "CC"
    EXECUTIVE_CHAIR_CAR = "EC"
    SECOND_SITTING = "2S"


def coerce_availability_state(value: AvailabilityState | str) -> AvailabilityState:
    """Coerce ``value`` to :class:`AvailabilityState` or raise."""
    if isinstance(value, AvailabilityState):
        return value
    if isinstance(value, str):
        candidate = value.strip().upper().replace(" ", "_")
        try:
            return AvailabilityState(candidate)
        except ValueError as exc:  # pragma: no cover - message is asserted in tests
            raise ValueError(
                f"unsupported availability state {value!r}; "
                f"expected one of {[s.value for s in AvailabilityState]}"
            ) from exc
    raise ValueError(f"availability state must be a string, got {type(value).__name__}")


def coerce_travel_class(value: TravelClass | str) -> TravelClass:
    """Coerce ``value`` to :class:`TravelClass` or raise."""
    if isinstance(value, TravelClass):
        return value
    if isinstance(value, str):
        candidate = value.strip().upper()
        try:
            return TravelClass(candidate)
        except ValueError as exc:
            raise ValueError(
                f"unsupported travel class {value!r}; "
                f"expected one of {[c.value for c in TravelClass]}"
            ) from exc
    raise ValueError(f"travel class must be a string, got {type(value).__name__}")
