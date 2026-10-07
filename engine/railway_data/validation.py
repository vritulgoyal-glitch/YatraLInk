"""Focused validation for the normalized railway data contract (Phase 2A).

Every invariant the contract enforces lives here, so the models stay declarative
and a provider adapter can read, in one place, exactly what it must satisfy.
The helpers deliberately reuse Phase 1's primitives (:mod:`engine.time_utils`,
:mod:`engine.enums`) instead of re-deciding them:

* clock times stay **naive, minute-granular wall-clock times**, because the time
  published in a timetable is a wall-clock time and nothing here may silently
  shift it;
* a stop's moment in absolute time is ``service date + day_offset`` *localised*
  to a station timezone -- never *converted* -- so a stop after midnight keeps
  both its calendar day and its published clock time;
* facts that really are instants (``fetched_at``, ``updated_at``) must be
  timezone-aware; a naive one is rejected rather than assumed to be UTC.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime, time, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from engine import time_utils as tu
from engine.enums import AvailabilityState, coerce_availability_state
from engine.errors import DomainValidationError
from engine.models import normalize_station_code

__all__ = [
    "MINUTES_PER_DAY",
    "as_tzinfo",
    "localise",
    "require_aware_datetime",
    "require_non_empty_text",
    "require_optional_count",
    "require_ordered_route",
    "require_paise",
    "require_schedule_date",
    "require_timezone_name",
    "require_unique_stations",
    "schedule_minutes",
    "validate_day_offsets",
    "validate_sequence",
    "validate_state_counters",
    "validate_stop_times",
]

MINUTES_PER_DAY = tu.MINUTES_PER_DAY

#: Counters each availability state is allowed to carry.  A state may only
#: describe itself: an ``UNKNOWN`` snapshot that also carried a seat count would
#: be an availability claim, and preventing exactly that is the point of the
#: contract's ``UNKNOWN`` rule.
_STATE_COUNTERS: dict[AvailabilityState, tuple[str, ...]] = {
    AvailabilityState.AVAILABLE: ("available_count",),
    AvailabilityState.RAC: ("rac_count",),
    AvailabilityState.WAITLIST: ("waitlist_value",),
    AvailabilityState.NOT_AVAILABLE: (),
    AvailabilityState.UNKNOWN: (),
}


def require_non_empty_text(value: object, label: str) -> str:
    """Return ``value`` stripped, rejecting anything that is not a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError(f"{label} must be a non-empty string, got {value!r}")
    return value.strip()


def require_schedule_date(value: object, label: str) -> date:
    """Validate a calendar date.

    A :class:`datetime.datetime` is rejected because it carries a time of day:
    a schedule date is a whole calendar day, and accepting a datetime here would
    invite a silent truncation.
    """
    return tu.require_date(value, label)


def require_aware_datetime(value: object, label: str) -> datetime:
    """Validate a timezone-aware datetime.

    ``fetched_at`` and ``updated_at`` describe *instants*; a naive datetime does
    not identify one, so it is rejected instead of being assumed to be UTC.
    """
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DomainValidationError(
            f"{label} must be a timezone-aware datetime, got {value!r}; an instant "
            "in time cannot be naive"
        )
    return value


def require_optional_count(value: object, label: str, *, minimum: int = 1) -> int | None:
    """Validate an optional integer counter; ``None`` means "not reported".

    ``bool`` is rejected along with every non-integer, and a reported counter
    must be at least ``minimum`` -- reporting "0 seats available" as a confirmed
    availability is a contradiction, not a fact.
    """
    if value is None:
        return None
    return tu.require_int(value, label, minimum=minimum)


def require_paise(value: object, label: str) -> int:
    """Validate a fare as a non-negative integer number of paise.

    Money is never a float and never a bool: ``12.0``, ``True`` and ``-1`` are
    all rejected rather than coerced, so a fare can always be summed exactly.
    """
    if isinstance(value, bool) or isinstance(value, float):
        raise DomainValidationError(
            f"{label} must be an int number of paise, got {type(value).__name__}; "
            "money must never be a float or a bool"
        )
    if not isinstance(value, int):
        raise DomainValidationError(
            f"{label} must be an int number of paise, got {type(value).__name__}"
        )
    if value < 0:
        raise DomainValidationError(f"{label} must be >= 0, got {value}")
    return value


def require_timezone_name(value: object, label: str = "timezone") -> str:
    """Validate an IANA timezone name such as ``"Asia/Kolkata"``."""
    name = require_non_empty_text(value, label)
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise DomainValidationError(f"{label} {name!r} is not a known IANA timezone") from exc
    return name


def as_tzinfo(value: tzinfo | str, label: str = "timezone") -> tzinfo:
    """Coerce a timezone name or an existing ``tzinfo`` into a ``tzinfo``."""
    if isinstance(value, tzinfo):
        return value
    return ZoneInfo(require_timezone_name(value, label))


def localise(day: date, clock: time, timezone: tzinfo | str) -> datetime:
    """Localise a wall-clock schedule time into an aware datetime.

    This is *localisation*, not conversion: the calendar day and the clock time
    are used exactly as given and merely labelled with the station's timezone.
    ``astimezone`` is never applied, so the published timetable time is preserved
    and a day offset is never silently absorbed into a timezone shift.
    """
    validated_day = require_schedule_date(day, "schedule day")
    validated_clock = tu.require_clock(clock, "schedule clock")
    return datetime.combine(validated_day, validated_clock, tzinfo=as_tzinfo(timezone))


def schedule_minutes(day_offset: object, clock: object, label: str = "stop") -> int:
    """Exact minutes from the service day's midnight for one stop.

    ``day_offset`` takes part in the arithmetic explicitly, so a stop at
    ``23:55`` on day 0 and a stop at ``00:10`` on day 1 can never be confused.
    """
    offset = tu.require_int(day_offset, f"{label} day_offset", minimum=0)
    validated_clock = tu.require_clock(clock, f"{label} clock")
    return offset * MINUTES_PER_DAY + tu.minutes_from_midnight(validated_clock)


def validate_sequence(
    sequences: Sequence[int] | Iterable[int], label: str = "stop"
) -> tuple[int, ...]:
    """Validate stop sequences: positive integers, unique, strictly increasing."""
    materialised = tuple(
        tu.require_int(sequence, f"{label} sequence", minimum=1) for sequence in sequences
    )
    for index in range(1, len(materialised)):
        previous, current = materialised[index - 1], materialised[index]
        if current == previous:
            raise DomainValidationError(f"{label} sequences must be unique, got {current} twice")
        if current < previous:
            raise DomainValidationError(
                f"{label} sequences must be strictly increasing, got {materialised}"
            )
    return materialised


def require_unique_stations(codes: Sequence[str], label: str = "route") -> tuple[str, ...]:
    """Reject a route that visits one station more than once."""
    materialised = tuple(codes)
    if len(set(materialised)) != len(materialised):
        raise DomainValidationError(f"{label} visits a station more than once: {materialised}")
    return materialised


def validate_day_offsets(offsets: Sequence[int], label: str = "stop") -> tuple[int, ...]:
    """Validate day offsets: non-negative integers that never decrease along a route."""
    materialised = tuple(
        tu.require_int(offset, f"{label} day_offset", minimum=0) for offset in offsets
    )
    for index in range(1, len(materialised)):
        if materialised[index] < materialised[index - 1]:
            raise DomainValidationError(
                f"{label} day offsets must never decrease, got {materialised}"
            )
    return materialised


def validate_stop_times(
    *,
    station_code: str,
    sequence: int,
    arrival: time | None,
    departure: time | None,
    day_offset: int,
    departure_day_offset: int | None,
    label: str = "stop",
) -> int:
    """Validate one stop's clock facts and return its effective ``departure_day_offset``.

    Rules enforced:

    * a stop must carry an arrival time, a departure time, or both;
    * both clocks must be naive and minute-granular (Phase 1 rule, reused);
    * ``departure_day_offset`` is ``day_offset`` or ``day_offset + 1`` -- the one
      documented way to express a halt that itself crosses midnight;
    * a departure may not precede its own arrival.
    """
    where = f"{label} {sequence} at {station_code}"
    if arrival is None and departure is None:
        raise DomainValidationError(
            f"{where} must have an arrival time, a departure time, or both"
        )
    if arrival is not None:
        tu.require_clock(arrival, f"{where} arrival")
    if departure is not None:
        tu.require_clock(departure, f"{where} departure")
    offset = tu.require_int(day_offset, f"{where} day_offset", minimum=0)

    if departure_day_offset is None:
        effective_departure_offset = offset
    else:
        effective_departure_offset = tu.require_int(
            departure_day_offset, f"{where} departure_day_offset", minimum=0
        )
        if effective_departure_offset not in (offset, offset + 1):
            raise DomainValidationError(
                f"{where}: departure_day_offset must equal day_offset ({offset}) or "
                f"day_offset + 1 ({offset + 1}), got {effective_departure_offset}"
            )

    if arrival is not None and departure is not None:
        arrival_minutes = schedule_minutes(offset, arrival, where)
        departure_minutes = schedule_minutes(effective_departure_offset, departure, where)
        if departure_minutes < arrival_minutes:
            raise DomainValidationError(
                f"{where}: departure {tu.format_clock(departure)} "
                f"(day +{effective_departure_offset}) precedes arrival "
                f"{tu.format_clock(arrival)} (day +{offset})"
            )
    return effective_departure_offset


def require_ordered_route(
    codes: Sequence[str], origin: str, destination: str, label: str = "route"
) -> tuple[int, int]:
    """Validate that ``origin`` occurs strictly before ``destination`` in ``codes``.

    Returns the two positions.  A reversed pair is rejected rather than swapped:
    the contract never reverses a route to make a request fit.
    """
    materialised = tuple(normalize_station_code(code) for code in codes)
    origin_code = normalize_station_code(origin)
    destination_code = normalize_station_code(destination)
    if origin_code == destination_code:
        raise DomainValidationError(
            f"{label} origin and destination must differ, got {origin_code} twice"
        )
    for code in (origin_code, destination_code):
        if code not in materialised:
            raise DomainValidationError(f"{label} does not contain station {code}")
    start, end = materialised.index(origin_code), materialised.index(destination_code)
    if start >= end:
        raise DomainValidationError(
            f"{label} visits {destination_code} before {origin_code}; a route is never reversed"
        )
    return start, end


def validate_state_counters(
    state: AvailabilityState | str,
    *,
    available_count: int | None = None,
    rac_count: int | None = None,
    waitlist_value: int | None = None,
    label: str = "availability snapshot",
) -> AvailabilityState:
    """Validate that the reported counters agree with the reported state.

    A counter is only meaningful for the state it describes (seats for
    ``AVAILABLE``, RAC places for ``RAC``, a waitlist position for ``WAITLIST``),
    and ``NOT_AVAILABLE`` / ``UNKNOWN`` may carry none at all.  Without this rule
    an adapter could return ``UNKNOWN`` plus ``available_count=1`` and quietly
    smuggle an availability claim through the contract, so a misplaced counter is
    an error rather than a hint.
    """
    resolved = coerce_availability_state(state)
    supplied = {
        "available_count": available_count,
        "rac_count": rac_count,
        "waitlist_value": waitlist_value,
    }
    permitted = _STATE_COUNTERS[resolved]
    for name, value in supplied.items():
        if value is None or name in permitted:
            continue
        allowed = ", ".join(permitted) if permitted else "no counter"
        raise DomainValidationError(
            f"{label} reports state {resolved.value} but also {name}; "
            f"{resolved.value} may carry {allowed}"
        )
    for name in permitted:
        require_optional_count(supplied[name], f"{label} {name}", minimum=1)
    return resolved
