"""Turning a structural candidate into a validated, priced journey option.

This module is the bridge between *structure* (:class:`engine.candidates.JourneyPlan`,
which knows only trains and times) and *reality* (availability, fares, transfer
risk, traveller constraints).

Two independent jobs live here:

1. **Assembly** — :func:`build_journey_option` converts a plan into a
   :class:`engine.models.JourneyOption`, attaching the availability and fare that
   the data source reported for each segment.  Missing data becomes
   :attr:`AvailabilityState.UNKNOWN` and a fare of ``None``; nothing is invented
   and no unknown is ever upgraded to AVAILABLE.
2. **Validation** — :func:`validate_journey_option` rejects a candidate that
   cannot be travelled or that violates the request.  Every rejection carries a
   machine-readable :class:`engine.enums.RejectionReason` together with a
   human-readable detail string, so a caller can report *why* a candidate was
   dropped instead of silently losing it.

Durations are always computed from absolute datetimes, and the journey's total
duration is ``final arrival - first departure`` — transfer waiting counts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

from engine import time_utils as tu
from engine.availability import DateAgnosticAvailability
from engine.candidates import JourneyPlan
from engine.config import SearchConfiguration
from engine.connections import validate_connection
from engine.enums import AvailabilityState, ConnectionKind, RejectionReason
from engine.errors import InvalidJourneyError
from engine.legs import RailLeg
from engine.models import ConnectionInfo, JourneyOption, JourneySegment, normalize_station_code
from engine.network import Timetable, TransferAllowance, lookup_transfer_allowance
from engine.provider import AvailabilityLookup
from engine.search import SearchRequest

__all__ = [
    "build_journey_option",
    "build_segment",
    "journey_id_for_plan",
    "segment_booking_date",
    "validate_journey_option",
]


def segment_booking_date(leg: RailLeg) -> date:
    """Calendar date on which a leg's reservation is booked.

    Each segment of a multi-segment journey is a separate reservation, and a
    reservation is always dated by the day the passenger actually boards.  For
    the first segment that is the requested travel date, because generation
    anchors the first leg to it; for later segments it is the day the connection
    departs, which is why an overnight connection looks up availability for the
    following calendar day.
    """
    return leg.departure.date()


def journey_id_for_plan(plan: JourneyPlan) -> str:
    """Stable, content-derived identifier for a plan.

    Derived only from the plan's structure, so the same candidate always receives
    the same id — across runs, processes and ranking passes.
    """
    digest = hashlib.sha1(repr(plan.signature).encode("utf-8")).hexdigest()[:12]
    return f"{plan.journey_type.value.lower()}-{digest}"


def build_segment(
    leg: RailLeg,
    *,
    request: SearchRequest,
    availability_book: AvailabilityLookup,
    timetable: Timetable,
    reservation_index: int,
) -> JourneySegment:
    """Convert one leg into a bookable segment, attaching reported inventory.

    Availability is looked up for the exact bookable unit
    (train / boarding date / class / quota / origin-destination).  When the data
    source has no record, the segment is ``UNKNOWN`` with an unknown fare — the
    engine reports the gap rather than filling it.
    """
    booking_date = segment_booking_date(leg)
    record = availability_book.lookup(
        train_number=leg.train_number,
        origin_station_code=leg.origin_station_code,
        destination_station_code=leg.destination_station_code,
        travel_date=booking_date,
        travel_class=request.travel_class,
        quota=request.quota,
    )
    if record is None:
        state = AvailabilityState.UNKNOWN
        fare = None
        seats = None
    else:
        state = record.state
        fare = record.fare
        seats = record.seats_available
        if isinstance(record, DateAgnosticAvailability):
            pass  # explicit snapshot-wide declaration; state/fare used as reported

    return JourneySegment(
        train_number=leg.train_number,
        train_name=leg.train.name,
        origin_station_code=leg.origin_station_code,
        destination_station_code=leg.destination_station_code,
        departure=leg.departure,
        arrival=leg.arrival,
        duration_minutes=leg.duration_minutes,
        availability=state,
        fare=fare,
        seats_available=seats,
        origin_station_name=timetable.station_name(leg.origin_station_code),
        destination_station_name=timetable.station_name(leg.destination_station_code),
        reservation_index=reservation_index,
        boarding_stop_sequence=leg.boarding_sequence,
        alighting_stop_sequence=leg.alighting_sequence,
    )


def build_journey_option(
    plan: JourneyPlan,
    *,
    request: SearchRequest,
    timetable: Timetable,
    availability_book: AvailabilityLookup,
    config: SearchConfiguration,
    allowances: tuple[TransferAllowance, ...] = (),
) -> JourneyOption:
    """Assemble a validated :class:`engine.models.JourneyOption` from a plan.

    Raises :class:`engine.errors.InvalidJourneyError` — carrying a
    :class:`engine.enums.RejectionReason` — when the plan cannot be turned into a
    travellable journey (impossible connection, missing station in the network,
    request constraint violated).
    """
    if len(plan.segments) > config.max_segments:
        raise InvalidJourneyError(
            f"plan has {len(plan.segments)} segments, the engine allows at most "
            f"{config.max_segments}",
            reason=RejectionReason.TOO_MANY_SEGMENTS,
        )

    segments = tuple(
        build_segment(
            leg,
            request=request,
            availability_book=availability_book,
            timetable=timetable,
            reservation_index=index,
        )
        for index, leg in enumerate(plan.segments)
    )

    connections = []
    for index, (previous, following) in enumerate(zip(segments, segments[1:], strict=False)):
        allowance = None
        if previous.destination_station_code != following.origin_station_code:
            allowance = lookup_transfer_allowance(
                previous.destination_station_code,
                following.origin_station_code,
                allowances,
            )
        connections.append(
            validate_connection(
                index=index,
                previous=previous,
                following=following,
                config=config,
                allowance=allowance,
            )
        )

    option = JourneyOption(
        journey_id=journey_id_for_plan(plan),
        journey_type=plan.journey_type,
        origin_station_code=plan.origin_station_code,
        destination_station_code=plan.destination_station_code,
        segments=segments,
        connections=tuple(connections),
    )
    validate_journey_option(option, request=request, config=config)
    return option


def validate_journey_option(
    option: JourneyOption,
    *,
    request: SearchRequest,
    config: SearchConfiguration,
) -> None:
    """Validate a fully assembled journey against the request and journey rules.

    Structural validation happens in the model constructors; this function adds
    the *journey-level* rules:

    * the journey must start at the request origin and end at the request
      destination;
    * segments must be contiguous in station order and strictly increasing in
      time (a connection may not be re-entered backwards);
    * a station change is allowed **only** when a valid
      :class:`engine.models.ConnectionInfo` in ``option.connections`` proves that
      the transfer is permissible — i.e. the data declared a transfer allowance
      for that ordered station pair.  The record is the proof: without it, two
      different station codes are never assumed to be the same place;
    * departure must not precede the traveller's earliest departure;
    * arrival must not exceed the traveller's latest arrival;
    * train changes may not exceed the request's or the engine's limit;
    * every required segment must be travellable: a ``NOT_AVAILABLE`` segment
      makes the journey impossible, and ``UNKNOWN`` is never treated as confirmed.
    """
    if normalize_station_code(option.origin_station_code) != request.origin:
        raise InvalidJourneyError(
            f"journey starts at {option.origin_station_code}, request starts at {request.origin}",
            reason=RejectionReason.MALFORMED,
        )
    if normalize_station_code(option.destination_station_code) != request.destination:
        raise InvalidJourneyError(
            f"journey ends at {option.destination_station_code}, request ends at "
            f"{request.destination}",
            reason=RejectionReason.MALFORMED,
        )

    for index, segment in enumerate(option.segments):
        for code in (segment.origin_station_code, segment.destination_station_code):
            if not _station_exists(option, code):  # pragma: no cover - defensive
                raise InvalidJourneyError(
                    f"segment {index} references station {code}, which is unknown",
                    reason=RejectionReason.MALFORMED,
                )
        if segment.origin_station_code == segment.destination_station_code:
            raise InvalidJourneyError(
                f"segment {index} on train {segment.train_number} has identical origin "
                f"and destination ({segment.origin_station_code})",
                reason=RejectionReason.MALFORMED,
            )

    for index, (previous, following) in enumerate(
        zip(option.segments, option.segments[1:], strict=False)
    ):
        same_station = previous.destination_station_code == following.origin_station_code
        if not same_station:
            proof = _station_change_proof(option, index, previous, following)
            if proof is None:
                raise InvalidJourneyError(
                    f"segment {index + 1} starts at {following.origin_station_code} but "
                    f"segment {index} ends at {previous.destination_station_code}, and "
                    "no valid transfer record proves that station change is allowed",
                    reason=RejectionReason.MALFORMED,
                )
            if not proof.valid:  # pragma: no cover - build already rejects invalid
                raise InvalidJourneyError(
                    f"station change {proof.from_station_code} -> "
                    f"{proof.to_station_code} is not valid: {proof.note}",
                    reason=RejectionReason.INVALID_CONNECTION,
                )
        if following.departure < previous.arrival:
            raise InvalidJourneyError(
                f"segment {index + 1} departs {following.departure.isoformat()} before "
                f"segment {index} arrives {previous.arrival.isoformat()}",
                reason=RejectionReason.BROKEN_CHRONOLOGY,
            )

    earliest = request.earliest_departure_datetime
    if earliest is not None and option.departure < earliest:
        raise InvalidJourneyError(
            f"journey departs {option.departure.isoformat()}, before the requested "
            f"earliest departure {earliest.isoformat()}",
            reason=RejectionReason.BEFORE_EARLIEST_DEPARTURE,
        )
    if request.latest_arrival is not None and option.arrival > request.latest_arrival:
        raise InvalidJourneyError(
            f"journey arrives {option.arrival.isoformat()}, after the requested latest "
            f"arrival {request.latest_arrival.isoformat()}",
            reason=RejectionReason.AFTER_LATEST_ARRIVAL,
        )

    allowed_changes = config.max_train_changes
    if request.max_train_changes is not None:
        allowed_changes = min(allowed_changes, request.max_train_changes)
    if option.train_changes > allowed_changes:
        raise InvalidJourneyError(
            f"journey needs {option.train_changes} train change(s), the limit is {allowed_changes}",
            reason=RejectionReason.TOO_MANY_TRAIN_CHANGES,
        )

    if option.total_duration_minutes <= 0:
        raise InvalidJourneyError(
            f"journey duration must be positive, got {option.total_duration_minutes} minute(s)",
            reason=RejectionReason.BROKEN_CHRONOLOGY,
        )

    if not config.include_not_available_journeys and any(
        segment.availability is AvailabilityState.NOT_AVAILABLE for segment in option.segments
    ):
        blocked = [
            f"{segment.train_number} {segment.origin_station_code}"
            f"->{segment.destination_station_code}"
            for segment in option.segments
            if segment.availability is AvailabilityState.NOT_AVAILABLE
        ]
        raise InvalidJourneyError(
            "cannot be travelled: NOT_AVAILABLE on " + ", ".join(blocked),
            reason=RejectionReason.SEGMENT_NOT_AVAILABLE,
        )


def _station_exists(option: JourneyOption, station_code: str) -> bool:
    """A station is known when the journey itself names it.

    The journey's own stations are the endpoints of its segments.  A station
    change that the data proves (for example NDLS -> DEL, backed by a declared
    transfer allowance) deliberately makes the next segment's origin different
    from the previous segment's destination, so membership is tested against the
    segment endpoints and the station path together.
    """
    code = normalize_station_code(station_code)
    for segment in option.segments:
        if code in (segment.origin_station_code, segment.destination_station_code):
            return True
    return code in option.station_path


def _station_change_proof(
    option: JourneyOption,
    index: int,
    previous: JourneySegment,
    following: JourneySegment,
) -> ConnectionInfo | None:
    """The transfer record that proves a station change at ``index`` is permitted.

    Returns ``None`` when no record matches the two segments' station and time
    facts, which means the station change is unproven and must be rejected.
    """
    for connection in option.connections:
        if connection.index != index:
            continue
        if connection.kind is not ConnectionKind.CROSS_STATION_TRANSFER:
            continue
        if connection.from_station_code != previous.destination_station_code:
            continue
        if connection.to_station_code != following.origin_station_code:
            continue
        if connection.arrival != previous.arrival:
            continue
        if connection.departure != following.departure:
            continue
        return connection
    return None


@dataclass(frozen=True, slots=True)
class JourneyValidationOutcome:
    """Result of validating a plan: either an option or an explicit rejection."""

    option: JourneyOption | None
    reason: RejectionReason | None
    detail: str

    @property
    def accepted(self) -> bool:
        return self.option is not None


def try_build_journey_option(
    plan: JourneyPlan,
    *,
    request: SearchRequest,
    timetable: Timetable,
    availability_book: AvailabilityLookup,
    config: SearchConfiguration,
    allowances: tuple[TransferAllowance, ...] = (),
) -> JourneyValidationOutcome:
    """Validate a plan without raising, returning the reason on rejection.

    Only :class:`engine.errors.DomainValidationError` (which covers every
    documented rejection, including :class:`engine.errors.InvalidJourneyError`)
    is converted into an outcome.  Any other exception is a defect and is allowed
    to propagate rather than be swallowed.
    """
    from engine.errors import DomainValidationError

    try:
        option = build_journey_option(
            plan,
            request=request,
            timetable=timetable,
            availability_book=availability_book,
            config=config,
            allowances=allowances,
        )
    except DomainValidationError as error:
        reason = getattr(error, "reason", None) or RejectionReason.MALFORMED
        return JourneyValidationOutcome(option=None, reason=reason, detail=str(error))
    return JourneyValidationOutcome(option=option, reason=None, detail="accepted")


def total_waiting_minutes(option: JourneyOption) -> int:
    """Minutes spent waiting between segments.

    Elapsed journey time minus time actually spent on board: the difference the
    specification warns about when it says not to sum segment durations.
    """
    return tu.minutes_between(option.departure, option.arrival) - option.total_travel_minutes
