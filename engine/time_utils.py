"""Deterministic date/time helpers for railway schedules.

A railway timetable expresses each stop as a wall-clock time plus a **day
offset** relative to the day on which the train's route starts.  A stop is
converted to a comparable naive local ``datetime`` by combining three facts:

1. the travel date (or, for later segments, the date of that train's occurrence),
2. the stop's ``day_offset``,
3. the stop's clock time.

Phase 1 deliberately avoids timezones and daylight-saving complexity: Indian
railway schedules are published in a single local railway time, and mixing
timezone conversions into journey arithmetic would add failure modes without
adding correctness.  Times are therefore *naive local* datetimes and all
comparisons are exact.

Clock times are minute-granular.  A time carrying non-zero seconds or
microseconds is rejected rather than silently truncated, so that every duration
computed by the engine is an exact whole number of minutes.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from engine.errors import DomainValidationError, InvalidJourneyError

__all__ = [
    "MINUTES_PER_DAY",
    "add_days",
    "anchor_date_for_boarding",
    "boarding_anchor_date",
    "combine",
    "duration_minutes",
    "first_departure_on_or_after",
    "format_clock",
    "minutes_between",
    "minutes_from_midnight",
    "parse_clock",
    "stop_datetime",
]

MINUTES_PER_DAY = 24 * 60


def require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    """Validate that ``value`` is an int (not a bool) within ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DomainValidationError(f"{label} must be an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise DomainValidationError(f"{label} must be >= {minimum}, got {value}")
    return value


def require_clock(value: object, label: str) -> time:
    """Validate that ``value`` is a naive, minute-granular :class:`datetime.time`."""
    if not isinstance(value, time):
        raise DomainValidationError(f"{label} must be a datetime.time, got {type(value).__name__}")
    if value.tzinfo is not None:
        raise DomainValidationError(f"{label} must be a naive local time, not timezone aware")
    if value.second or value.microsecond:
        raise DomainValidationError(
            f"{label} {value.isoformat()} has sub-minute precision; "
            "railway schedule times must be whole minutes"
        )
    return value


def require_date(value: object, label: str) -> date:
    """Validate that ``value`` is a :class:`datetime.date` (not a datetime)."""
    if isinstance(value, datetime) or not isinstance(value, date):
        raise DomainValidationError(f"{label} must be a datetime.date, got {type(value).__name__}")
    return value


def require_datetime(value: object, label: str) -> datetime:
    """Validate that ``value`` is a naive :class:`datetime.datetime`."""
    if not isinstance(value, datetime):
        raise DomainValidationError(
            f"{label} must be a datetime.datetime, got {type(value).__name__}"
        )
    if value.tzinfo is not None:
        raise DomainValidationError(f"{label} must be a naive local datetime, not timezone aware")
    return value


def parse_clock(text: str) -> time:
    """Parse ``"HH:MM"`` (or ``"HH:MM:SS"`` with zero seconds) into a time."""
    if not isinstance(text, str):
        raise DomainValidationError(f"clock text must be a string, got {type(text).__name__}")
    cleaned = text.strip()
    if not cleaned:
        raise DomainValidationError("clock text must not be empty")
    for pattern in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(cleaned, pattern).time()
        except ValueError:
            continue
        return require_clock(parsed, f"clock {text!r}")
    raise DomainValidationError(
        f"cannot parse clock text {text!r}; expected HH:MM or HH:MM:SS (24-hour)"
    )


def format_clock(value: time) -> str:
    """Format a time as ``"HH:MM"``."""
    require_clock(value, "clock")
    return f"{value.hour:02d}:{value.minute:02d}"


def minutes_from_midnight(value: time) -> int:
    """Minutes elapsed since midnight (exact, because times are minute-granular)."""
    require_clock(value, "clock")
    return value.hour * 60 + value.minute


def combine(day: date, clock: time) -> datetime:
    """Combine a naive date and a minute-granular clock into a datetime."""
    require_date(day, "day")
    require_clock(clock, "clock")
    return datetime.combine(day, clock)


def add_days(day: date, days: int) -> date:
    """Return ``day`` shifted by ``days`` (may be negative)."""
    require_date(day, "day")
    require_int(days, "days")
    return day + timedelta(days=days)


def stop_datetime(anchor_date: date, day_offset: int, clock: time) -> datetime:
    """Absolute datetime of a stop, given the train occurrence's anchor date.

    ``anchor_date`` is the date on which the train's ``day_offset == 0`` starts.
    """
    require_int(day_offset, "day_offset", minimum=0)
    return combine(add_days(anchor_date, day_offset), clock)


def anchor_date_for_boarding(travel_date: date, departure_day_offset: int) -> date:
    """Anchor date that makes a train board at ``travel_date``.

    For the first segment of a journey, the passenger must depart the requested
    origin on the requested travel date, whatever the stop's day offset is
    inside the train's own route.
    """
    require_date(travel_date, "travel_date")
    require_int(departure_day_offset, "departure_day_offset", minimum=0)
    return travel_date - timedelta(days=departure_day_offset)


def first_departure_on_or_after(threshold: datetime, clock: time) -> datetime:
    """Earliest datetime with time ``clock`` that is >= ``threshold``.

    Used to pick the earliest feasible occurrence of the next train when
    validating a connection: services are assumed to run daily, so the answer is
    either the same calendar day or the next one.
    """
    require_datetime(threshold, "threshold")
    require_clock(clock, "clock")
    candidate = combine(threshold.date(), clock)
    if candidate < threshold:
        candidate = candidate + timedelta(days=1)
    return candidate


def boarding_anchor_date(
    threshold: datetime, departure_clock: time, departure_day_offset: int
) -> date:
    """Anchor date of the earliest occurrence that departs at or after ``threshold``.

    Train services are modelled as running daily.  Given the moment a passenger
    becomes available at a station, this returns the anchor date (the date on
    which the train's ``day_offset == 0`` starts) of the earliest occurrence of
    the train that departs no earlier than that moment.

    The returned anchor is *not* simply ``threshold.date()``: a stop carrying a
    day offset of 1 or more boards on the calendar day after its anchor, so the
    anchor must be shifted back.  This is what keeps overnight and multi-day
    routes correct.
    """
    require_datetime(threshold, "threshold")
    require_clock(departure_clock, "departure_clock")
    require_int(departure_day_offset, "departure_day_offset", minimum=0)
    departure = first_departure_on_or_after(threshold, departure_clock)
    return departure.date() - timedelta(days=departure_day_offset)


def minutes_between(start: datetime, end: datetime) -> int:
    """Exact whole minutes from ``start`` to ``end`` (signed)."""
    require_datetime(start, "start")
    require_datetime(end, "end")
    return int((end - start) // timedelta(minutes=1))


def duration_minutes(start: datetime, end: datetime, *, context: str = "segment") -> int:
    """Positive elapsed minutes from ``start`` to ``end``.

    A non-positive duration is a defect, not a result: the engine never returns
    a negative travel time, and it never pretends a zero-length leg is travel.
    """
    elapsed = minutes_between(start, end)
    if elapsed <= 0:
        raise InvalidJourneyError(
            f"{context} duration must be positive, got {elapsed} minute(s) "
            f"({start.isoformat()} -> {end.isoformat()})"
        )
    return elapsed
