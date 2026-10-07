"""The search request: what the traveller asks the engine to find.

A request is fully described by the traveller's input plus a small number of
optional constraints.  Validation happens on construction, so no algorithm ever
has to defend against a half-valid request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time

from engine import time_utils as tu
from engine.config import MAX_PASSENGERS, MIN_PASSENGERS
from engine.enums import TravelClass, coerce_travel_class
from engine.errors import DomainValidationError, SearchRequestError
from engine.models import normalize_station_code

__all__ = ["DEFAULT_QUOTA", "SearchRequest"]

#: Default reservation quota when the traveller does not choose one.
DEFAULT_QUOTA = "GENERAL"


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """A validated journey search.

    :param origin: origin station code (normalised on construction).
    :param destination: destination station code.
    :param travel_date: the calendar date on which the journey must **start**;
        the passenger departs ``origin`` on this date.
    :param passengers: number of travellers, 1-6.
    :param travel_class: reservation class, defaults to 3A.
    :param quota: reservation quota, defaults to ``GENERAL``.
    :param earliest_departure: optional clock time on ``travel_date`` before which
        the journey may not start.
    :param latest_arrival: optional absolute deadline by which the journey must
        be complete.
    :param max_train_changes: optional per-request limit, never above the engine
        maximum of 2.
    """

    origin: str
    destination: str
    travel_date: date
    passengers: int = 1
    travel_class: TravelClass = TravelClass.AC_3_TIER
    quota: str = DEFAULT_QUOTA
    earliest_departure: time | None = None
    latest_arrival: datetime | None = None
    max_train_changes: int | None = None

    def __post_init__(self) -> None:
        """Validate the request, reporting every problem as :class:`SearchRequestError`.

        Domain errors raised by the model/time helpers are re-raised as
        ``SearchRequestError`` so that a caller only has to handle one error type
        for "the request you gave me is not usable".
        """
        try:
            self._validate()
        except SearchRequestError:
            raise
        except DomainValidationError as error:
            raise SearchRequestError(str(error)) from error

    def _validate(self) -> None:
        origin = normalize_station_code(self.origin)
        destination = normalize_station_code(self.destination)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "destination", destination)
        if origin == destination:
            raise SearchRequestError(f"origin and destination must differ, both are {origin}")

        tu.require_date(self.travel_date, "SearchRequest.travel_date")

        tu.require_int(self.passengers, "SearchRequest.passengers", minimum=MIN_PASSENGERS)
        if self.passengers > MAX_PASSENGERS:
            raise SearchRequestError(
                f"passengers must be between {MIN_PASSENGERS} and {MAX_PASSENGERS}, "
                f"got {self.passengers}"
            )

        object.__setattr__(self, "travel_class", coerce_travel_class(self.travel_class))

        if not isinstance(self.quota, str) or not self.quota.strip():
            raise SearchRequestError("quota must be a non-empty string")
        object.__setattr__(self, "quota", self.quota.strip().upper())

        if self.earliest_departure is not None:
            tu.require_clock(self.earliest_departure, "SearchRequest.earliest_departure")
        if self.latest_arrival is not None:
            tu.require_datetime(self.latest_arrival, "SearchRequest.latest_arrival")
            if self.latest_arrival <= tu.combine(self.travel_date, time(0, 0)):
                raise SearchRequestError(
                    "latest_arrival must be after the start of the travel date"
                )
            if (
                self.earliest_departure is not None
                and self.latest_arrival <= self.earliest_departure_datetime  # type: ignore[operator]
            ):
                raise SearchRequestError("latest_arrival must be after earliest_departure")
        if self.max_train_changes is not None:
            tu.require_int(self.max_train_changes, "SearchRequest.max_train_changes", minimum=0)
            if self.max_train_changes > 2:
                raise SearchRequestError(
                    "Phase 1 supports at most 2 train changes, got "
                    f"max_train_changes={self.max_train_changes}"
                )

    # ------------------------------------------------------------------
    # derived values
    # ------------------------------------------------------------------
    @property
    def earliest_departure_datetime(self) -> datetime | None:
        """``earliest_departure`` resolved onto ``travel_date``."""
        if self.earliest_departure is None:
            return None
        return tu.combine(self.travel_date, self.earliest_departure)

    @property
    def start_of_travel_date(self) -> datetime:
        """Midnight of the travel date, the lower bound for generation."""
        return tu.combine(self.travel_date, time(0, 0))

    def describe(self) -> str:
        """Short human-readable summary, useful in logs and reports."""
        parts = [
            f"{self.origin}->{self.destination}",
            self.travel_date.isoformat(),
            self.travel_class.value,
            f"{self.passengers}pax",
            self.quota,
        ]
        if self.earliest_departure is not None:
            parts.append(f"after {tu.format_clock(self.earliest_departure)}")
        if self.latest_arrival is not None:
            parts.append(f"by {self.latest_arrival.isoformat()}")
        return " ".join(parts)

    def to_json(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "travel_date": self.travel_date.isoformat(),
            "passengers": self.passengers,
            "travel_class": self.travel_class.value,
            "quota": self.quota,
            "earliest_departure": (
                tu.format_clock(self.earliest_departure) if self.earliest_departure else None
            ),
            "latest_arrival": (self.latest_arrival.isoformat() if self.latest_arrival else None),
            "max_train_changes": self.max_train_changes,
        }
