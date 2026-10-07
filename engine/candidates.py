"""Candidate generation: direct, same-train split and connecting journeys.

Three generators, one shared shape
----------------------------------
Each generator returns :class:`JourneyPlan` objects: an ordered tuple of
:class:`engine.legs.RailLeg` values plus the journey type.  A plan contains no
availability, no fare and no score — it is pure **structure**.  Turning a plan
into a priced, availability-checked :class:`engine.models.JourneyOption` (and
rejecting it when it is not valid) is the job of :mod:`engine.journey`.

That split is what makes the engine testable: structure can be asserted
independently of data, and data can be swapped without touching the algorithms.

Depth limit
-----------
Phase 1 supports at most 2 train changes (3 segments).  The connecting generator
is a bounded depth-first search, **not** a general shortest-path search, and it
cannot exceed the configured segment limit.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from engine.config import SearchConfiguration
from engine.enums import JourneyType
from engine.errors import CandidateLimitError, DomainValidationError
from engine.legs import RailLeg, boarding_anchor, legs_from_station
from engine.network import Timetable, TransferAllowance
from engine.search import SearchRequest

__all__ = [
    "JourneyPlan",
    "generate_all_plans",
    "generate_connecting_plans",
    "generate_direct_plans",
    "generate_same_train_split_plans",
]


@dataclass(frozen=True, slots=True)
class JourneyPlan:
    """A structural candidate: the legs and the journey type, nothing priced."""

    journey_type: JourneyType
    segments: tuple[RailLeg, ...]
    station_links: frozenset[tuple[str, str]] = frozenset()

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments:
            raise DomainValidationError("a JourneyPlan needs at least one segment")
        object.__setattr__(self, "segments", segments)
        for previous, following in zip(segments, segments[1:], strict=False):
            contiguous = previous.destination_station_code == following.origin_station_code
            linked = (previous.destination_station_code, following.origin_station_code) in (
                self.station_links
            )
            if not (contiguous or linked):
                raise DomainValidationError(
                    "plan segments are neither contiguous nor joined by a declared "
                    f"transfer allowance: {previous.describe()} cannot be followed by "
                    f"{following.describe()}"
                )
            if following.departure < previous.arrival:
                raise DomainValidationError(
                    "plan segments are not chronological: "
                    f"{following.describe()} departs before {previous.describe()} arrives"
                )
        if (
            self.journey_type is JourneyType.SAME_TRAIN_SPLIT
            and len({segment.train_number for segment in segments}) != 1
        ):
            raise DomainValidationError(
                "a SAME_TRAIN_SPLIT plan must stay on one train, got "
                f"{[segment.train_number for segment in segments]}"
            )

    @property
    def origin_station_code(self) -> str:
        return self.segments[0].origin_station_code

    @property
    def destination_station_code(self) -> str:
        return self.segments[-1].destination_station_code

    @property
    def departure(self) -> datetime:
        return self.segments[0].departure

    @property
    def arrival(self) -> datetime:
        return self.segments[-1].arrival

    @property
    def train_changes(self) -> int:
        """Physical train transitions in the plan."""
        return sum(
            1
            for previous, following in zip(self.segments, self.segments[1:], strict=False)
            if previous.train_number != following.train_number
        )

    @property
    def signature(self) -> tuple[object, ...]:
        """Ordered train/station structure, used to describe a rejection."""
        return (
            self.journey_type.value,
            tuple(
                (
                    segment.train_number,
                    segment.origin_station_code,
                    segment.destination_station_code,
                )
                for segment in self.segments
            ),
            self.departure,
            self.arrival,
        )

    def describe(self) -> str:
        return " | ".join(segment.describe() for segment in self.segments)


def _first_threshold(request: SearchRequest) -> datetime:
    """Moment from which the first leg may depart (start of the travel date or later)."""
    earliest = request.earliest_departure_datetime
    return earliest if earliest is not None else request.start_of_travel_date


def generate_direct_plans(
    timetable: Timetable,
    request: SearchRequest,
    config: SearchConfiguration,
) -> tuple[JourneyPlan, ...]:
    """All single-leg candidates from origin to destination.

    A direct candidate exists only when one train's route contains the origin
    *before* the destination.  A train that travels the opposite way is never
    reversed to satisfy the request.
    """
    if not config.enable_direct:
        return ()
    legs = legs_from_station(
        timetable,
        request.origin,
        _first_threshold(request),
        config=config,
        deadline=request.latest_arrival,
        destination_filter=request.destination,
    )
    return tuple(JourneyPlan(JourneyType.DIRECT, (leg,)) for leg in legs)


def generate_same_train_split_plans(
    timetable: Timetable,
    request: SearchRequest,
    config: SearchConfiguration,
) -> tuple[JourneyPlan, ...]:
    """Same-train split candidates: two or more reservations, no train change.

    For every train whose route visits the origin before the destination, each
    ordered combination of intermediate stops of the required size splits the
    ride into separate reservations.  Combinations are enumerated in route order,
    so output is deterministic and duplicate-free.
    """
    if not config.enable_same_train_split:
        return ()
    max_segments = min(config.same_train_split_max_segments, config.max_segments)
    if max_segments < 2:
        return ()

    threshold = _first_threshold(request)
    plans: list[JourneyPlan] = []

    for train in timetable.trains_between(request.origin, request.destination):
        origin_index = train.stop_index(request.origin)
        destination_index = train.stop_index(request.destination)
        anchor = boarding_anchor(train, origin_index, threshold)
        intermediate = list(range(origin_index + 1, destination_index))

        for segment_count in range(2, max_segments + 1):
            for combo in itertools.combinations(intermediate, segment_count - 1):
                boundaries = (origin_index, *combo, destination_index)
                legs = tuple(
                    RailLeg(
                        train=train,
                        origin_index=boundaries[position],
                        destination_index=boundaries[position + 1],
                        anchor_date=anchor,
                    )
                    for position in range(len(boundaries) - 1)
                )
                if request.latest_arrival is not None and legs[-1].arrival > request.latest_arrival:
                    continue
                plans.append(JourneyPlan(JourneyType.SAME_TRAIN_SPLIT, legs))
                if len(plans) > config.max_candidates_per_type:
                    raise CandidateLimitError(
                        "same-train split generation exceeded "
                        f"max_candidates_per_type={config.max_candidates_per_type}"
                    )

    return tuple(plans)


def generate_connecting_plans(
    timetable: Timetable,
    request: SearchRequest,
    config: SearchConfiguration,
    allowances: Iterable[TransferAllowance] = (),
) -> tuple[JourneyPlan, ...]:
    """Connecting candidates: one or two train changes, at most three segments.

    The search is a bounded depth-first expansion:

    * expand only from the station where the previous leg ended;
    * never continue on the train just used (that is the same-train split case);
    * never re-use a train already used in the plan (prevents cycles);
    * never return to a station already visited (prevents cycles);
    * stop expanding once the configured segment limit is reached.

    Every expansion necessarily changes train, so a connecting plan always has at
    least one change.  Single-train journeys are produced by the direct and
    same-train split generators instead.

    When ``allow_cross_station_transfers`` is enabled, expansion additionally
    considers stations that the railway data explicitly links to the arrival
    station through a :class:`engine.network.TransferAllowance`.  Without such an
    allowance (or with the switch off) a station change is never considered, so
    the engine never assumes two station codes are the same place.
    """
    if not config.enable_connecting:
        return ()

    max_segments = min(config.max_segments, config.max_train_changes + 1)
    if max_segments < 2:
        return ()

    allowance_list = tuple(allowances)
    links = _station_links(allowance_list, config)
    threshold = _first_threshold(request)
    deadline = request.latest_arrival
    plans: list[JourneyPlan] = []

    def expansion_stations(arrival_station: str) -> tuple[str, ...]:
        """Stations from which the next leg may be boarded, in deterministic order.

        The request destination is never used as a transfer point: boarding there
        could only produce a journey that leaves the destination and comes back,
        which is exactly the kind of circular path the engine must not offer.
        """
        linked = tuple(
            destination
            for source, destination in sorted(links)
            if source == arrival_station
            and destination != arrival_station
            and destination != request.destination
        )
        if arrival_station == request.destination:
            return ()
        return (arrival_station, *linked)

    def expand(
        legs: tuple[RailLeg, ...],
        used_trains: frozenset[str],
        visited_stations: frozenset[str],
    ) -> None:
        last = legs[-1]
        for boarding_station in expansion_stations(last.destination_station_code):
            candidates = legs_from_station(
                timetable,
                boarding_station,
                last.arrival,
                config=config,
                deadline=deadline,
                excluded_trains=used_trains,
                excluded_stations=visited_stations,
            )
            for leg in candidates:
                plan_legs = (*legs, leg)
                reaches_destination = leg.destination_station_code == request.destination
                if reaches_destination:
                    plans.append(
                        JourneyPlan(JourneyType.CONNECTING, plan_legs, station_links=links)
                    )
                    if len(plans) > config.max_candidates_per_type:
                        raise CandidateLimitError(
                            "connecting generation exceeded "
                            f"max_candidates_per_type={config.max_candidates_per_type}"
                        )
                    continue
                if len(plan_legs) >= max_segments:
                    continue
                expand(
                    plan_legs,
                    used_trains | {leg.train_number},
                    visited_stations | {leg.destination_station_code},
                )

    first_legs = legs_from_station(
        timetable,
        request.origin,
        threshold,
        config=config,
        deadline=deadline,
        destination_filter=None,
    )
    for first_leg in first_legs:
        # A single leg that already reaches the destination is a direct journey.
        if first_leg.destination_station_code == request.destination:
            continue
        expand(
            (first_leg,),
            frozenset({first_leg.train_number}),
            frozenset({request.origin, first_leg.destination_station_code}),
        )

    return tuple(plans)


def _station_links(
    allowances: tuple[TransferAllowance, ...], config: SearchConfiguration
) -> frozenset[tuple[str, str]]:
    """Ordered station pairs the data explicitly allows as a transfer."""
    if not config.allow_cross_station_transfers:
        return frozenset()
    return frozenset(allowance.key for allowance in allowances)


def generate_all_plans(
    timetable: Timetable,
    request: SearchRequest,
    config: SearchConfiguration,
    allowances: Iterable[TransferAllowance] = (),
) -> tuple[JourneyPlan, ...]:
    """Run every enabled generator and return the plans in a fixed order.

    Order: direct first, then same-train splits, then connecting journeys — the
    order in which a traveller would prefer to consider them.  Each generator's
    own output order is deterministic, so the concatenation is deterministic too.
    """
    timetable.station(request.origin)
    timetable.station(request.destination)
    plans = (
        *generate_direct_plans(timetable, request, config),
        *generate_same_train_split_plans(timetable, request, config),
        *generate_connecting_plans(timetable, request, config, allowances),
    )
    return plans
