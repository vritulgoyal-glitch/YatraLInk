"""Deterministic, explainable journey ranking.

Design goals (all enforced by construction)
-------------------------------------------
1. **No AI, no randomness, no clock, no I/O.**  Ranking is a pure function of the
   candidate set and the configured weights.
2. **No floating point.**  Every score is an integer in ``0..1000``, so ties are
   exact and reproducible across platforms.
3. **Explainable.**  A journey's score is a list of named components; each carries
   its weight, its normalised 0-1000 sub-score, its weighted contribution, and a
   human-readable detail string.
4. **Total order.**  Candidates are sorted by score and then by an explicit,
   documented chain of tiebreakers ending in a stable structural key, so the
   output order never depends on dict ordering or input order.

Score model — **higher is better**, range ``0..1000``
------------------------------------------------------
Seven components are scored on the same 0-1000 scale (1000 is always best) and
combined as a weighted mean::

    total = sum(weight_i * subscore_i) // sum(weight_i)

Each sub-score is a *relative* comparison against the achievable range inside
this candidate set (see :func:`normalise`), which keeps the scale meaningful even
though fare and duration have no absolute maximum.

============================  =================================================
Component                     Meaning of 1000
============================  =================================================
``availability``              every segment AVAILABLE
``train_changes``             direct (no train change)
``fare``                      lowest total fare of the candidate set
``duration``                  shortest elapsed time of the candidate set
``connection_risk``           all connections SAFE
``separate_reservations``     a single reservation
``station_change``            no station change at any transfer
============================  =================================================

``UNKNOWN`` availability and an unknown fare are never scored as if they were
good: ``UNKNOWN`` sits on the documented availability ladder, and an unknown fare
receives a zero fare contribution (sub-score 0), so the fare component never
rewards an unpriced journey.  The overall ranking is still determined by the
complete weighted score, so other components may legitimately outweigh the fare
component.

Configurable behaviour
----------------------
Weights live in :class:`engine.config.RankingWeights`, so "optimise for time" or
"optimise for price" is a configuration change, not a code change.  Weights are
relative: only their ratios matter because the total is normalised by their sum.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from engine.config import RankingWeights
from engine.enums import (
    AVAILABILITY_DESIRABILITY,
    AvailabilityState,
    ConnectionRisk,
    JourneyType,
)
from engine.errors import DomainValidationError
from engine.models import JourneyOption

__all__ = [
    "JourneyScore",
    "RankedJourney",
    "ScoreComponent",
    "ScoreContext",
    "ScoreScale",
    "build_score_context",
    "compare_journeys",
    "rank_journeys",
    "score_journey",
    "sort_journeys",
]

#: Sub-scores are integers on this scale; higher is better.
MAX_SUBSCORE = 1000

#: Preferred journey type order, used as a documented tiebreaker.
_TYPE_PREFERENCE: dict[JourneyType, int] = {
    JourneyType.DIRECT: 0,
    JourneyType.SAME_TRAIN_SPLIT: 1,
    JourneyType.CONNECTING: 2,
}


class ScoreScale:
    """Names of the scored dimensions (kept as constants to avoid typos)."""

    AVAILABILITY = "availability"
    TRAIN_CHANGES = "train_changes"
    FARE = "fare"
    DURATION = "duration"
    CONNECTION_RISK = "connection_risk"
    SEPARATE_RESERVATIONS = "separate_reservations"
    STATION_CHANGE = "station_change"

    ORDER: tuple[str, ...] = (
        AVAILABILITY,
        TRAIN_CHANGES,
        FARE,
        DURATION,
        CONNECTION_RISK,
        SEPARATE_RESERVATIONS,
        STATION_CHANGE,
    )


@dataclass(frozen=True, slots=True)
class ScoreContext:
    """Reference values that make relative sub-scores meaningful.

    Built from the candidate set by :func:`build_score_context`: the best and
    worst observed fare, duration, train changes, reservation count and station
    changes.  Using the observed range means the scale adapts to the search
    without ever making the comparison arbitrary.
    """

    min_duration_minutes: int
    max_duration_minutes: int
    min_fare_paise: int | None
    max_fare_paise: int | None
    min_train_changes: int
    max_train_changes: int
    min_reservations: int
    max_reservations: int
    min_station_changes: int
    max_station_changes: int

    def __post_init__(self) -> None:
        if self.min_duration_minutes < 0 or self.max_duration_minutes < self.min_duration_minutes:
            raise DomainValidationError("ScoreContext duration range is inconsistent")
        if self.max_train_changes < self.min_train_changes:
            raise DomainValidationError("ScoreContext train-change range is inconsistent")
        if self.max_reservations < self.min_reservations:
            raise DomainValidationError("ScoreContext reservation range is inconsistent")
        if self.max_station_changes < self.min_station_changes:
            raise DomainValidationError("ScoreContext station-change range is inconsistent")


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    """One scored dimension of a journey, with its contribution."""

    name: str
    weight: int
    subscore: int
    contribution: int
    detail: str

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "weight": self.weight,
            "subscore": self.subscore,
            "contribution": self.contribution,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class JourneyScore:
    """The full, explainable score of one journey."""

    journey_id: str
    total: int
    components: tuple[ScoreComponent, ...]
    weights: RankingWeights

    def __post_init__(self) -> None:
        if not 0 <= self.total <= MAX_SUBSCORE:
            raise DomainValidationError(
                f"total score must be within 0..{MAX_SUBSCORE}, got {self.total}"
            )

    def component(self, name: str) -> ScoreComponent:
        """Look up one component by name."""
        for item in self.components:
            if item.name == name:
                return item
        raise KeyError(f"no score component named {name!r}")

    @property
    def as_dict(self) -> dict[str, int]:
        """Sub-scores by dimension name (for API payloads and reports)."""
        return {item.name: item.subscore for item in self.components}

    def explain(self) -> str:
        """One-line, human-readable breakdown, ordered by weight."""
        parts = [
            f"{item.name}={item.subscore} (w{item.weight})"
            for item in sorted(self.components, key=lambda c: (-c.weight, c.name))
        ]
        return f"total={self.total}/1000 | " + ", ".join(parts)

    def to_json(self) -> dict[str, object]:
        return {
            "journey_id": self.journey_id,
            "total": self.total,
            "scale": MAX_SUBSCORE,
            "weights": self.weights.to_json(),
            "components": [item.to_json() for item in self.components],
        }


@dataclass(frozen=True, slots=True)
class RankedJourney:
    """A journey together with its score and its 1-based rank."""

    rank: int
    journey: JourneyOption
    score: JourneyScore

    @property
    def journey_id(self) -> str:
        return self.journey.journey_id

    def to_json(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score.to_json(),
            "journey": {
                "journey_id": self.journey.journey_id,
                "journey_type": self.journey.journey_type.value,
                "origin_station_code": self.journey.origin_station_code,
                "destination_station_code": self.journey.destination_station_code,
                "departure": self.journey.departure.isoformat(),
                "arrival": self.journey.arrival.isoformat(),
                "total_duration_minutes": self.journey.total_duration_minutes,
                "total_travel_minutes": self.journey.total_travel_minutes,
                "total_transfer_minutes": self.journey.total_transfer_minutes,
                "train_changes": self.journey.train_changes,
                "reservation_count": self.journey.reservation_count,
                "availability": self.journey.availability.value,
                "risk": self.journey.risk.value,
                "total_fare": (
                    self.journey.total_fare.to_json() if self.journey.total_fare else None
                ),
                "segments": [segment.to_json() for segment in self.journey.segments],
                "connections": [connection.to_json() for connection in self.journey.connections],
            },
        }


def normalise(value: int, *, best: int, worst: int, higher_is_better: bool) -> int:
    """Map ``value`` onto ``0..1000`` deterministically.

    ``best`` is the value that deserves :data:`MAX_SUBSCORE` and ``worst`` the
    value that deserves ``0``; everything between is scaled with integer
    arithmetic, so the result is exact and platform-independent.

    ``higher_is_better=True`` means a *larger* value deserves the higher score,
    in which case ``best`` must be the larger of the two reference values.  For
    fares and durations — where smaller is better — pass
    ``higher_is_better=False`` with ``best`` set to the minimum observed value.

    When ``best == worst`` the whole candidate set shares one value, so there is
    nothing to discriminate on: the component scores :data:`MAX_SUBSCORE` for
    everyone and neither rewards nor penalises any candidate.
    """
    if higher_is_better:
        span = best - worst
        offset = best - value
    else:
        span = worst - best
        offset = value - best
    if span <= 0:
        return MAX_SUBSCORE
    clamped = max(0, min(span, offset))
    return MAX_SUBSCORE - (clamped * MAX_SUBSCORE) // span


def build_score_context(journeys: Sequence[JourneyOption]) -> ScoreContext:
    """Derive the reference ranges used to normalise sub-scores."""
    if not journeys:
        raise DomainValidationError("cannot build a score context from zero journeys")

    durations = [journey.total_duration_minutes for journey in journeys]
    changes = [journey.train_changes for journey in journeys]
    reservations = [journey.reservation_count for journey in journeys]
    station_changes = [
        sum(1 for connection in journey.connections if connection.station_change)
        for journey in journeys
    ]
    fares = [
        fare.amount_paise
        for fare in (journey.total_fare for journey in journeys)
        if fare is not None
    ]
    return ScoreContext(
        min_duration_minutes=min(durations),
        max_duration_minutes=max(durations),
        min_fare_paise=min(fares) if fares else None,
        max_fare_paise=max(fares) if fares else None,
        min_train_changes=min(changes),
        max_train_changes=max(changes),
        min_reservations=min(reservations),
        max_reservations=max(reservations),
        min_station_changes=min(station_changes),
        max_station_changes=max(station_changes),
    )


def _availability_subscore(journey: JourneyOption) -> tuple[int, str]:
    worst = max(
        (segment.availability for segment in journey.segments),
        key=lambda state: AVAILABILITY_DESIRABILITY[state],
    )
    ladder = AVAILABILITY_DESIRABILITY[worst]
    span = max(AVAILABILITY_DESIRABILITY.values())
    subscore = MAX_SUBSCORE - (ladder * MAX_SUBSCORE) // span
    detail = f"weakest segment {worst.value}"
    if worst is AvailabilityState.UNKNOWN:
        detail += " (unknown is never treated as confirmed)"
    return subscore, detail


def _risk_subscore(risk: ConnectionRisk) -> tuple[int, str]:
    ladder = risk.desirability
    span = max(2, 1)
    subscore = MAX_SUBSCORE - (ladder * MAX_SUBSCORE) // span
    return subscore, f"worst connection risk {risk.value}"


def score_journey(
    journey: JourneyOption,
    *,
    context: ScoreContext,
    weights: RankingWeights,
) -> JourneyScore:
    """Score one journey against a candidate-set context.

    The returned components are ordered as in :attr:`ScoreScale.ORDER`, and
    ``total`` is the weight-normalised mean of the components.
    """
    availability_subscore, availability_detail = _availability_subscore(journey)
    risk = journey.risk
    risk_subscore, risk_detail = _risk_subscore(risk)

    duration_subscore = normalise(
        journey.total_duration_minutes,
        best=context.min_duration_minutes,
        worst=context.max_duration_minutes,
        higher_is_better=False,
    )
    changes_subscore = normalise(
        journey.train_changes,
        best=context.min_train_changes,
        worst=context.max_train_changes,
        higher_is_better=False,
    )
    reservations_subscore = normalise(
        journey.reservation_count,
        best=context.min_reservations,
        worst=context.max_reservations,
        higher_is_better=False,
    )
    station_changes = sum(1 for connection in journey.connections if connection.station_change)
    station_change_subscore = normalise(
        station_changes,
        best=context.min_station_changes,
        worst=context.max_station_changes,
        higher_is_better=False,
    )

    total_fare = journey.total_fare
    if total_fare is None or context.min_fare_paise is None or context.max_fare_paise is None:
        # An unknown fare contributes nothing: zero fare sub-score.  The overall
        # ranking is still decided by the complete weighted score.
        fare_subscore = 0
        fare_detail = (
            "fare unknown; scored at the worst level because an unpriced journey "
            "must not outrank a priced one"
        )
    else:
        fare_subscore = normalise(
            total_fare.amount_paise,
            best=context.min_fare_paise,
            worst=context.max_fare_paise,
            higher_is_better=False,
        )
        fare_detail = f"total {total_fare.format()}"

    components = (
        ScoreComponent(
            name=ScoreScale.AVAILABILITY,
            weight=weights.availability,
            subscore=availability_subscore,
            contribution=weights.availability * availability_subscore,
            detail=availability_detail,
        ),
        ScoreComponent(
            name=ScoreScale.TRAIN_CHANGES,
            weight=weights.train_changes,
            subscore=changes_subscore,
            contribution=weights.train_changes * changes_subscore,
            detail=f"{journey.train_changes} train change(s)",
        ),
        ScoreComponent(
            name=ScoreScale.FARE,
            weight=weights.fare,
            subscore=fare_subscore,
            contribution=weights.fare * fare_subscore,
            detail=fare_detail,
        ),
        ScoreComponent(
            name=ScoreScale.DURATION,
            weight=weights.duration,
            subscore=duration_subscore,
            contribution=weights.duration * duration_subscore,
            detail=(
                f"{journey.total_duration_minutes} min total "
                f"({journey.total_travel_minutes} min travelling, "
                f"{journey.total_transfer_minutes} min transferring)"
            ),
        ),
        ScoreComponent(
            name=ScoreScale.CONNECTION_RISK,
            weight=weights.connection_risk,
            subscore=risk_subscore,
            contribution=weights.connection_risk * risk_subscore,
            detail=risk_detail,
        ),
        ScoreComponent(
            name=ScoreScale.SEPARATE_RESERVATIONS,
            weight=weights.separate_reservations,
            subscore=reservations_subscore,
            contribution=weights.separate_reservations * reservations_subscore,
            detail=(
                f"{journey.reservation_count} reservation(s)"
                + (
                    "; separate tickets required even though the train does not change"
                    if journey.journey_type is JourneyType.SAME_TRAIN_SPLIT
                    else ""
                )
            ),
        ),
        ScoreComponent(
            name=ScoreScale.STATION_CHANGE,
            weight=weights.station_change,
            subscore=station_change_subscore,
            contribution=weights.station_change * station_change_subscore,
            detail=f"{station_changes} station change(s)",
        ),
    )

    total = sum(item.contribution for item in components) // weights.total_weight
    return JourneyScore(
        journey_id=journey.journey_id,
        total=total,
        components=tuple(sorted(components, key=lambda item: ScoreScale.ORDER.index(item.name))),
        weights=weights,
    )


def compare_journeys(left: JourneyOption, right: JourneyOption) -> int:
    """Documented tiebreaker chain for two journeys with equal scores.

    Order of preference (best first): DIRECT, then SAME_TRAIN_SPLIT, then
    CONNECTING; then SAFE < TIGHT risk; then better availability; then fewer train
    changes; then lower fare (unknown fare last); then shorter duration; then
    later departure; finally the stable structural key.  Returns a negative number
    when ``left`` should be ranked before ``right``.
    """
    if _TYPE_PREFERENCE[left.journey_type] != _TYPE_PREFERENCE[right.journey_type]:
        return _TYPE_PREFERENCE[left.journey_type] - _TYPE_PREFERENCE[right.journey_type]

    if left.risk.desirability != right.risk.desirability:
        return left.risk.desirability - right.risk.desirability

    left_availability = AVAILABILITY_DESIRABILITY[left.availability]
    right_availability = AVAILABILITY_DESIRABILITY[right.availability]
    if left_availability != right_availability:
        return left_availability - right_availability

    if left.train_changes != right.train_changes:
        return left.train_changes - right.train_changes

    left_fare = left.total_fare.amount_paise if left.total_fare else None
    right_fare = right.total_fare.amount_paise if right.total_fare else None
    if left_fare != right_fare:
        if left_fare is None:
            return 1
        if right_fare is None:
            return -1
        return left_fare - right_fare

    if left.total_duration_minutes != right.total_duration_minutes:
        return left.total_duration_minutes - right.total_duration_minutes

    if left.departure != right.departure:
        return -1 if left.departure > right.departure else 1

    left_key = left.stable_sort_key
    right_key = right.stable_sort_key
    if left_key != right_key:
        return -1 if left_key < right_key else 1
    return 0


def sort_journeys(
    journeys: Iterable[JourneyOption],
    *,
    weights: RankingWeights,
) -> tuple[RankedJourney, ...]:
    """Rank journeys from best to worst, deterministically.

    The result depends only on the *set* of candidates and the weights: shuffling
    the input cannot change the order, and two identical candidates always appear
    in the same relative order.
    """
    materialised = tuple(journeys)
    if not materialised:
        return ()
    context = build_score_context(materialised)
    scored = [
        (journey, score_journey(journey, context=context, weights=weights))
        for journey in materialised
    ]

    import functools

    def compare(
        left: tuple[JourneyOption, JourneyScore], right: tuple[JourneyOption, JourneyScore]
    ) -> int:
        if left[1].total != right[1].total:
            return -1 if left[1].total > right[1].total else 1
        return compare_journeys(left[0], right[0])

    ordered = sorted(scored, key=functools.cmp_to_key(compare))
    return tuple(
        RankedJourney(rank=position, journey=journey, score=score)
        for position, (journey, score) in enumerate(ordered, start=1)
    )


def rank_journeys(
    journeys: Iterable[JourneyOption],
    *,
    weights: RankingWeights,
) -> tuple[JourneyOption, ...]:
    """Convenience wrapper returning just the journeys, best first."""
    return tuple(ranked.journey for ranked in sort_journeys(journeys, weights=weights))
