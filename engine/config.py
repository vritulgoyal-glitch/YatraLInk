"""Engine configuration: search limits, connection rules and ranking weights.

Every rule the engine applies is configurable here so that product decisions can
be tuned without editing algorithms.  Defaults are deliberately conservative and
mirror the Phase 1 specification:

* minimum connection buffer: 30 minutes
* a connection whose buffer is within 30 minutes is ``TIGHT``, above it ``SAFE``
* maximum 2 train changes, therefore at most 3 segments
* different-station transfers are **disabled** unless the data explicitly
  provides a transfer allowance for that station pair
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.errors import DomainValidationError
from engine.time_utils import require_int

__all__ = ["MAX_PASSENGERS", "MIN_PASSENGERS", "RankingWeights", "SearchConfiguration"]

#: Passenger-count bounds from the product requirements document.
MIN_PASSENGERS = 1
MAX_PASSENGERS = 6


@dataclass(frozen=True, slots=True)
class RankingWeights:
    """Relative importance of each ranking dimension.

    Weights are non-negative integers; their absolute size is irrelevant because
    the total score is normalised by their sum.  Every component score is an
    integer in ``0..1000`` (higher is better), so the total score is also an
    integer in ``0..1000`` and comparison is exact — no floating point is used
    anywhere in ranking.

    The default profile makes **availability dominant** (40 of 100 points), then
    values convenience (train changes 15, connection risk 10), then cost and time
    (fare 15, duration 15), with two small tie-shaping terms.  This matches the
    product principle that a feasible, low-risk journey should outrank a
    marginally cheaper or faster but less certain one.
    """

    availability: int = 40
    train_changes: int = 15
    fare: int = 15
    duration: int = 15
    connection_risk: int = 10
    separate_reservations: int = 3
    station_change: int = 2

    def __post_init__(self) -> None:
        for name in (
            "availability",
            "train_changes",
            "fare",
            "duration",
            "connection_risk",
            "separate_reservations",
            "station_change",
        ):
            require_int(getattr(self, name), f"RankingWeights.{name}", minimum=0)
        if self.total_weight <= 0:
            raise DomainValidationError("RankingWeights must not be all zero")

    @property
    def total_weight(self) -> int:
        return (
            self.availability
            + self.train_changes
            + self.fare
            + self.duration
            + self.connection_risk
            + self.separate_reservations
            + self.station_change
        )

    def to_json(self) -> dict[str, int]:
        return {
            "availability": self.availability,
            "train_changes": self.train_changes,
            "fare": self.fare,
            "duration": self.duration,
            "connection_risk": self.connection_risk,
            "separate_reservations": self.separate_reservations,
            "station_change": self.station_change,
            "total_weight": self.total_weight,
        }


@dataclass(frozen=True, slots=True)
class SearchConfiguration:
    """Deterministic rules used to generate, validate and rank journeys."""

    # --- connection rules ------------------------------------------------
    #: Minimum transfer time between two different trains at the same station.
    minimum_connection_minutes: int = 30
    #: A valid connection whose spare buffer is <= this is TIGHT; above it SAFE.
    tight_connection_max_buffer_minutes: int = 30
    #: Default requirement for a different-station transfer when the data source
    #: provides an allowance but with a smaller value than this.
    cross_station_minimum_minutes: int = 90
    #: Master switch: different-station transfers are never accepted unless this
    #: is enabled *and* the data explicitly allows the specific station pair.
    allow_cross_station_transfers: bool = False

    # --- structural limits ----------------------------------------------
    max_train_changes: int = 2
    max_segments: int = 3
    #: Same-train splits with more than this many reservations are not generated.
    same_train_split_max_segments: int = 2
    #: Safety valve so a large timetable cannot explode the candidate set.
    max_candidates_per_type: int = 4000

    # --- feature switches ------------------------------------------------
    enable_direct: bool = True
    enable_same_train_split: bool = True
    enable_connecting: bool = True
    #: When False (default) a journey with a NOT_AVAILABLE segment is rejected
    #: before ranking, because it cannot be travelled.
    include_not_available_journeys: bool = False

    ranking_weights: RankingWeights = field(default_factory=RankingWeights)

    def __post_init__(self) -> None:
        require_int(self.minimum_connection_minutes, "minimum_connection_minutes", minimum=0)
        require_int(
            self.tight_connection_max_buffer_minutes,
            "tight_connection_max_buffer_minutes",
            minimum=0,
        )
        require_int(self.cross_station_minimum_minutes, "cross_station_minimum_minutes", minimum=0)
        require_int(self.max_train_changes, "max_train_changes", minimum=0)
        if self.max_train_changes > 2:
            raise DomainValidationError(
                "Phase 1 supports at most 2 train changes, got "
                f"max_train_changes={self.max_train_changes}"
            )
        require_int(self.max_segments, "max_segments", minimum=1)
        if self.max_segments > 3:
            raise DomainValidationError(
                f"Phase 1 supports at most 3 segments, got max_segments={self.max_segments}"
            )
        if self.max_train_changes + 1 > self.max_segments:
            raise DomainValidationError(
                "max_segments must be at least max_train_changes + 1 "
                f"(got max_segments={self.max_segments}, "
                f"max_train_changes={self.max_train_changes})"
            )
        require_int(
            self.same_train_split_max_segments,
            "same_train_split_max_segments",
            minimum=2,
        )
        if self.same_train_split_max_segments > self.max_segments:
            raise DomainValidationError(
                "same_train_split_max_segments must not exceed max_segments"
            )
        require_int(self.max_candidates_per_type, "max_candidates_per_type", minimum=1)
        if not isinstance(self.ranking_weights, RankingWeights):
            raise DomainValidationError("ranking_weights must be a RankingWeights instance")

    @property
    def required_buffers(self) -> dict[str, int]:
        """Requirement per transfer kind, useful for diagnostics and reports."""
        return {
            "SAME_TRAIN": 0,
            "CROSS_TRAIN_SAME_STATION": self.minimum_connection_minutes,
            "CROSS_STATION_TRANSFER": self.cross_station_minimum_minutes,
        }

    def to_json(self) -> dict[str, object]:
        return {
            "minimum_connection_minutes": self.minimum_connection_minutes,
            "tight_connection_max_buffer_minutes": self.tight_connection_max_buffer_minutes,
            "cross_station_minimum_minutes": self.cross_station_minimum_minutes,
            "allow_cross_station_transfers": self.allow_cross_station_transfers,
            "max_train_changes": self.max_train_changes,
            "max_segments": self.max_segments,
            "same_train_split_max_segments": self.same_train_split_max_segments,
            "max_candidates_per_type": self.max_candidates_per_type,
            "enable_direct": self.enable_direct,
            "enable_same_train_split": self.enable_same_train_split,
            "enable_connecting": self.enable_connecting,
            "include_not_available_journeys": self.include_not_available_journeys,
            "ranking_weights": self.ranking_weights.to_json(),
        }


#: Default configuration used when a caller does not supply one.
DEFAULT_CONFIGURATION = SearchConfiguration()
