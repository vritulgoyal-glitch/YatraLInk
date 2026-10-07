"""Connection validation and deterministic transfer-risk classification.

This module owns the single most error-prone part of journey generation: given
the arrival of one segment and the departure of the next, decide whether the
transfer is possible and how risky it is.

Rules (all configurable via :class:`engine.config.SearchConfiguration`)
---------------------------------------------------------------------

``SAME_TRAIN`` — the passenger never leaves the train
    Required transfer time is **zero**; the next leg may depart the moment the
    previous one arrives.  A same-train continuation is never risky in the sense
    of missing a train, so it classifies as ``SAFE``.

``CROSS_TRAIN_SAME_STATION`` — a real change of trains at one station
    Required transfer time is ``minimum_connection_minutes`` (default 30).
    ``buffer = transfer - required``:

    * ``buffer < 0``  -> ``INVALID`` (the connection cannot be made)
    * ``0 <= buffer <= tight_connection_max_buffer_minutes`` -> ``TIGHT``
    * ``buffer > tight_connection_max_buffer_minutes`` -> ``SAFE``

    Note that a connection of *exactly* the minimum (buffer ``0``) is valid but
    ``TIGHT``, matching the worked example in the specification.

``CROSS_STATION_TRANSFER`` — change of trains *and* of station
    Only ever possible when the railway data explicitly declares a
    :class:`engine.network.TransferAllowance` for that ordered station pair **and**
    ``allow_cross_station_transfers`` is enabled.  Otherwise the transfer is
    ``INVALID`` — the engine never assumes two station codes are the same place.
    When allowed, the requirement is ``max(allowance.minimum_minutes,
    cross_station_minimum_minutes)`` — the explicit allowance and the global
    cross-station minimum are *both* respected — and the risk is capped at
    ``TIGHT``: a station change is never ``SAFE``.  Without an explicit
    allowance the transfer stays ``INVALID`` even when
    ``allow_cross_station_transfers`` is enabled.

All arithmetic uses real :class:`datetime.datetime` values, never raw clock
times, so midnight-crossing and multi-day connections are handled exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from engine import time_utils as tu
from engine.config import SearchConfiguration
from engine.enums import ConnectionKind, ConnectionRisk, RejectionReason
from engine.errors import DomainValidationError, InvalidJourneyError
from engine.models import ConnectionInfo, JourneySegment
from engine.network import TransferAllowance

__all__ = [
    "ConnectionAssessment",
    "assess_connection",
    "classify_connection_kind",
    "connection_risk_for_buffer",
    "describe_connection_rule",
    "validate_connection",
]


@dataclass(frozen=True, slots=True)
class ConnectionAssessment:
    """The deterministic verdict on one transfer, before it becomes a journey fact."""

    kind: ConnectionKind
    transfer_minutes: int
    required_minutes: int
    buffer_minutes: int
    valid: bool
    risk: ConnectionRisk
    note: str = ""

    def __post_init__(self) -> None:
        if self.valid and self.risk is ConnectionRisk.INVALID:
            raise DomainValidationError(
                "an accepted connection cannot be classified as INVALID risk"
            )
        if not self.valid and self.risk is not ConnectionRisk.INVALID:
            raise DomainValidationError("a rejected connection must be classified as INVALID risk")


def classify_connection_kind(previous: JourneySegment, following: JourneySegment) -> ConnectionKind:
    """Classify the transfer between two consecutive segments."""
    if previous.train_number == following.train_number:
        return ConnectionKind.SAME_TRAIN
    if previous.destination_station_code == following.origin_station_code:
        return ConnectionKind.CROSS_TRAIN_SAME_STATION
    return ConnectionKind.CROSS_STATION_TRANSFER


def connection_risk_for_buffer(
    buffer_minutes: int, *, tight_connection_max_buffer_minutes: int
) -> ConnectionRisk:
    """Classify risk from the spare buffer after the minimum requirement."""
    if buffer_minutes < 0:
        return ConnectionRisk.INVALID
    if buffer_minutes <= tight_connection_max_buffer_minutes:
        return ConnectionRisk.TIGHT
    return ConnectionRisk.SAFE


def required_connection_minutes(
    kind: ConnectionKind,
    *,
    config: SearchConfiguration,
    allowance: TransferAllowance | None,
) -> int:
    """Minutes the transfer requires under ``config`` for a transfer of ``kind``.

    For a ``CROSS_STATION_TRANSFER`` the requirement is
    ``max(allowance.minimum_minutes, config.cross_station_minimum_minutes)``:
    the explicit allowance and the global cross-station minimum are both
    respected.  A different-station transfer *without* an explicit allowance is
    invalid by contract, so this helper raises
    :class:`engine.errors.DomainValidationError` instead of inventing a
    requirement for a transfer that can never be accepted.
    """
    if kind is ConnectionKind.SAME_TRAIN:
        return 0
    if kind is ConnectionKind.CROSS_TRAIN_SAME_STATION:
        return config.minimum_connection_minutes
    if allowance is None:
        raise DomainValidationError(
            "a different-station transfer requires an explicit TransferAllowance "
            "from the data source; without one the transfer is invalid"
        )
    return max(allowance.minimum_minutes, config.cross_station_minimum_minutes)


def assess_connection(
    *,
    kind: ConnectionKind,
    arrival: datetime,
    departure: datetime,
    config: SearchConfiguration,
    allowance: TransferAllowance | None = None,
) -> ConnectionAssessment:
    """Assess one transfer.

    :param arrival: arrival datetime of the previous segment.
    :param departure: departure datetime of the following segment.
    :param allowance: explicit data-source allowance for a station change, if any.
    """
    tu.require_datetime(arrival, "connection arrival")
    tu.require_datetime(departure, "connection departure")

    transfer = tu.minutes_between(arrival, departure)
    if kind is ConnectionKind.CROSS_STATION_TRANSFER and allowance is None:
        # No explicit allowance: the transfer is invalid by contract, however
        # generous the gap is.  Report the global cross-station minimum as the
        # requirement that no allowance satisfied, then reject below.
        required = config.cross_station_minimum_minutes
    else:
        required = required_connection_minutes(kind, config=config, allowance=allowance)
    buffer = transfer - required

    if kind is ConnectionKind.SAME_TRAIN:
        if transfer < 0:
            return ConnectionAssessment(
                kind=kind,
                transfer_minutes=transfer,
                required_minutes=0,
                buffer_minutes=buffer,
                valid=False,
                risk=ConnectionRisk.INVALID,
                note="the next reservation departs before the previous one arrives",
            )
        return ConnectionAssessment(
            kind=kind,
            transfer_minutes=transfer,
            required_minutes=0,
            buffer_minutes=buffer,
            valid=True,
            risk=ConnectionRisk.SAFE,
            note="same physical train: no transfer and no risk of missing the train",
        )

    if kind is ConnectionKind.CROSS_TRAIN_SAME_STATION:
        risk = connection_risk_for_buffer(
            buffer, tight_connection_max_buffer_minutes=config.tight_connection_max_buffer_minutes
        )
        if risk is not ConnectionRisk.INVALID:
            return ConnectionAssessment(
                kind=kind,
                transfer_minutes=transfer,
                required_minutes=required,
                buffer_minutes=buffer,
                valid=True,
                risk=risk,
                note=f"transfer at one station with {buffer} minute(s) of buffer",
            )
        return ConnectionAssessment(
            kind=kind,
            transfer_minutes=transfer,
            required_minutes=required,
            buffer_minutes=buffer,
            valid=False,
            risk=ConnectionRisk.INVALID,
            note=(f"transfer of {transfer} minute(s) is below the required {required} minute(s)"),
        )

    # --- CROSS_STATION_TRANSFER -----------------------------------------
    if not config.allow_cross_station_transfers:
        return ConnectionAssessment(
            kind=kind,
            transfer_minutes=transfer,
            required_minutes=required,
            buffer_minutes=buffer,
            valid=False,
            risk=ConnectionRisk.INVALID,
            note=(
                "different-station transfers are disabled; the data source must "
                "enable them explicitly"
            ),
        )
    if allowance is None:
        return ConnectionAssessment(
            kind=kind,
            transfer_minutes=transfer,
            required_minutes=required,
            buffer_minutes=buffer,
            valid=False,
            risk=ConnectionRisk.INVALID,
            note=("no transfer allowance declared by the data source for this station pair"),
        )
    risk = connection_risk_for_buffer(
        buffer, tight_connection_max_buffer_minutes=config.tight_connection_max_buffer_minutes
    )
    if risk is ConnectionRisk.INVALID:
        return ConnectionAssessment(
            kind=kind,
            transfer_minutes=transfer,
            required_minutes=required,
            buffer_minutes=buffer,
            valid=False,
            risk=ConnectionRisk.INVALID,
            note=(
                f"station-change transfer of {transfer} minute(s) is below the "
                f"required {required} minute(s)"
            ),
        )
    # A station change is never SAFE, even with a generous buffer.
    return ConnectionAssessment(
        kind=kind,
        transfer_minutes=transfer,
        required_minutes=required,
        buffer_minutes=buffer,
        valid=True,
        risk=ConnectionRisk.TIGHT,
        note=(
            "station change allowed by the data source; risk is capped at TIGHT "
            f"({buffer} minute(s) of buffer)"
        ),
    )


def validate_connection(
    *,
    index: int,
    previous: JourneySegment,
    following: JourneySegment,
    config: SearchConfiguration,
    allowance: TransferAllowance | None = None,
) -> ConnectionInfo:
    """Assess a transfer and return it as a :class:`engine.models.ConnectionInfo`.

    Raises :class:`engine.errors.InvalidJourneyError` when the transfer cannot be
    made.  A malformed or impossible candidate is rejected here, before it can
    reach ranking.
    """
    kind = classify_connection_kind(previous, following)
    assessment = assess_connection(
        kind=kind,
        arrival=previous.arrival,
        departure=following.departure,
        config=config,
        allowance=allowance,
    )
    if not assessment.valid:
        raise InvalidJourneyError(
            f"invalid connection {index}: {previous.destination_station_code} "
            f"({previous.arrival.isoformat()}) -> {following.origin_station_code} "
            f"({following.departure.isoformat()}): {assessment.note}",
            reason=RejectionReason.INVALID_CONNECTION,
        )
    return ConnectionInfo(
        index=index,
        kind=kind,
        arrival=previous.arrival,
        departure=following.departure,
        transfer_minutes=assessment.transfer_minutes,
        required_minutes=assessment.required_minutes,
        valid=True,
        risk=assessment.risk,
        from_station_code=previous.destination_station_code,
        to_station_code=following.origin_station_code,
        buffer_minutes=assessment.buffer_minutes,
        note=assessment.note,
    )


def describe_connection_rule(config: SearchConfiguration) -> str:
    """Human-readable description of the active connection rules."""
    return (
        f"same-train: 0 min required (always safe); "
        f"same-station change: {config.minimum_connection_minutes} min required, "
        f"TIGHT up to +{config.tight_connection_max_buffer_minutes} min buffer; "
        f"station change: "
        + (
            "allowed with an explicit allowance; the required time is "
            f"max(allowance, {config.cross_station_minimum_minutes} min) "
            "(risk capped at TIGHT)"
            if config.allow_cross_station_transfers
            else "not allowed unless explicitly enabled and declared by the data"
        )
    )
