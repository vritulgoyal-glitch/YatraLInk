"""The journey pipeline: generate, validate, deduplicate, rank.

This is the engine's public behaviour, and it is one deterministic function of
four inputs::

    generate_journeys(request, timetable, availability_book, configuration,
                      transfer_allowances) -> JourneyGenerationResult

Pipeline (each step is a separate, independently tested function)
----------------------------------------------------------------
1. the request is validated by :class:`engine.search.SearchRequest` on construction;
2. station codes are normalised by the model layer;
3. relevant train routes are selected from the timetable;
4. structural candidates are generated (direct, same-train split, connecting);
5. every candidate is assembled and validated, producing a priced
   :class:`engine.models.JourneyOption` **or** an explicit
   :class:`JourneyRejection` carrying a machine-readable reason;
6. candidates are deduplicated on journey type + ordered train/station structure +
   departure/arrival structure;
7. surviving candidates are ranked by :mod:`engine.ranking`;
8. the ranked list is returned, optionally truncated to ``limit``.

Nothing is dropped silently: the result carries every rejection with its reason
and detail string, so callers can audit candidate generation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from engine.candidates import JourneyPlan, generate_all_plans
from engine.config import SearchConfiguration
from engine.enums import AvailabilityState, JourneyType, RejectionReason
from engine.errors import CandidateLimitError, DomainValidationError
from engine.journey import build_journey_option
from engine.models import JourneyOption
from engine.network import Timetable, TransferAllowance
from engine.provider import AvailabilityLookup
from engine.ranking import RankedJourney, sort_journeys
from engine.search import SearchRequest

__all__ = [
    "CandidateLimitError",
    "JourneyGenerationResult",
    "JourneyRejection",
    "deduplicate_journeys",
    "generate_journey_options",
    "generate_journeys",
    "generate_journeys_for_network",
]


@dataclass(frozen=True, slots=True)
class JourneyRejection:
    """A candidate the engine generated but refused to return, and why."""

    signature: tuple[object, ...]
    reason: RejectionReason
    detail: str
    journey_type: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "reason": self.reason.value,
            "journey_type": self.journey_type,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class JourneyGenerationResult:
    """Everything the pipeline produced, including the audit trail."""

    request: SearchRequest
    configuration: SearchConfiguration
    journeys: tuple[RankedJourney, ...] = ()
    rejections: tuple[JourneyRejection, ...] = ()
    generated_plan_count: int = 0
    duplicate_count: int = 0
    candidate_count: int = 0
    type_counts: dict[str, int] = field(default_factory=dict)
    availability_counts: dict[str, int] = field(default_factory=dict)

    @property
    def options(self) -> tuple[JourneyOption, ...]:
        """The ranked journeys, best first."""
        return tuple(ranked.journey for ranked in self.journeys)

    @property
    def best(self) -> JourneyOption | None:
        """The highest ranked journey, or ``None`` when nothing was found."""
        return self.journeys[0].journey if self.journeys else None

    @property
    def rejection_counts(self) -> dict[str, int]:
        """Rejections grouped by reason, for diagnostics."""
        counts: dict[str, int] = {}
        for rejection in self.rejections:
            counts[rejection.reason.value] = counts.get(rejection.reason.value, 0) + 1
        return counts

    def to_json(self) -> dict[str, object]:
        return {
            "request": self.request.to_json(),
            "configuration": self.configuration.to_json(),
            "generated_plan_count": self.generated_plan_count,
            "candidate_count": self.candidate_count,
            "duplicate_count": self.duplicate_count,
            "journey_count": len(self.journeys),
            "type_counts": self.type_counts,
            "availability_counts": self.availability_counts,
            "rejection_counts": self.rejection_counts,
            "journeys": [ranked.to_json() for ranked in self.journeys],
            "rejections": [rejection.to_json() for rejection in self.rejections],
        }


def deduplicate_journeys(
    journeys: Iterable[JourneyOption],
) -> tuple[tuple[JourneyOption, ...], tuple[JourneyOption, ...]]:
    """Split candidates into unique journeys and duplicates.

    Two candidates are duplicates when :attr:`engine.models.JourneyOption.dedup_key`
    matches: same ordered train sequence, same ordered station sequence and the
    same departure/arrival structure.  The key includes the journey *type*, so a
    same-train split and a connecting journey over the same stations never
    collapse into one another.

    The first occurrence wins, which — because generation order is deterministic —
    makes the surviving set deterministic too.
    """
    seen: dict[tuple[object, ...], JourneyOption] = {}
    duplicates: list[JourneyOption] = []
    for journey in journeys:
        key = journey.dedup_key
        if key in seen:
            duplicates.append(journey)
        else:
            seen[key] = journey
    return tuple(seen.values()), tuple(duplicates)


def generate_journeys(
    search_request: SearchRequest,
    *,
    timetable: Timetable,
    availability_book: AvailabilityLookup,
    configuration: SearchConfiguration | None = None,
    transfer_allowances: Iterable[TransferAllowance] | None = None,
    limit: int | None = None,
) -> JourneyGenerationResult:
    """Run the full deterministic pipeline.

    :param search_request: the traveller's validated request.
    :param timetable: the railway snapshot to search.
    :param availability_book: the availability view — an in-memory
        :class:`engine.availability.AvailabilityBook`, or a
        :class:`engine.provider.ProviderAvailabilityBook` wrapping any
        :class:`engine.provider.AvailabilityProvider`.  The engine cannot tell
        the difference and never needs to.
    :param configuration: engine rules; defaults to :class:`SearchConfiguration`.
    :param transfer_allowances: explicit different-station transfer allowances.
        When ``None`` (the default) the allowances declared by ``timetable`` are
        used, so a caller holding a complete railway snapshot does not have to
        pass them separately.  An explicit empty tuple means "the data declares
        none", and then no station change is ever accepted.
    :param limit: optional cap on the number of journeys returned *after* ranking.
    :raises engine.errors.UnknownStationError: when a requested station is not in
        the timetable — a caller error, not an empty result.
    :raises engine.errors.CandidateLimitError: when a safety limit is exceeded.
    """
    if not isinstance(search_request, SearchRequest):
        raise DomainValidationError(
            f"search_request must be a SearchRequest, got {type(search_request).__name__}"
        )
    rules = configuration if configuration is not None else SearchConfiguration()
    allowances = (
        tuple(getattr(timetable, "transfer_allowances", ()))
        if transfer_allowances is None
        else tuple(transfer_allowances)
    )

    # Step 3: validate that both endpoints exist in this snapshot, so that a
    # missing station raises instead of quietly returning no journeys.
    timetable.station(search_request.origin)
    timetable.station(search_request.destination)

    plans: tuple[JourneyPlan, ...] = generate_all_plans(
        timetable, search_request, rules, allowances
    )

    options: list[JourneyOption] = []
    rejections: list[JourneyRejection] = []
    for plan in plans:
        try:
            option = build_journey_option(
                plan,
                request=search_request,
                timetable=timetable,
                availability_book=availability_book,
                config=rules,
                allowances=allowances,
            )
        except DomainValidationError as error:
            reason = getattr(error, "reason", None) or RejectionReason.MALFORMED
            rejections.append(
                JourneyRejection(
                    signature=plan.signature,
                    reason=reason,
                    detail=str(error),
                    journey_type=plan.journey_type.value,
                )
            )
            continue
        options.append(option)

    unique, duplicates = deduplicate_journeys(options)
    rejections.extend(
        JourneyRejection(
            signature=journey.dedup_key,
            reason=RejectionReason.DUPLICATE,
            detail="equivalent to an earlier candidate",
            journey_type=journey.journey_type.value,
        )
        for journey in duplicates
    )

    ranked = sort_journeys(unique, weights=rules.ranking_weights)
    if limit is not None:
        if limit < 0:
            raise DomainValidationError(f"limit must be >= 0, got {limit}")
        ranked = ranked[:limit]

    type_counts: dict[str, int] = {member.value: 0 for member in JourneyType}
    availability_counts: dict[str, int] = {member.value: 0 for member in AvailabilityState}
    for ranked_journey in ranked:
        type_counts[ranked_journey.journey.journey_type.value] += 1
        availability_counts[ranked_journey.journey.availability.value] += 1

    return JourneyGenerationResult(
        request=search_request,
        configuration=rules,
        journeys=ranked,
        rejections=tuple(rejections),
        generated_plan_count=len(plans),
        duplicate_count=len(duplicates),
        candidate_count=len(unique),
        type_counts=type_counts,
        availability_counts=availability_counts,
    )


def generate_journey_options(
    search_request: SearchRequest,
    *,
    timetable: Timetable,
    availability_book: AvailabilityLookup,
    configuration: SearchConfiguration | None = None,
    transfer_allowances: Iterable[TransferAllowance] | None = None,
    limit: int | None = None,
) -> tuple[JourneyOption, ...]:
    """Generate ranked journeys and return only the journey options.

    This is the conceptual interface from the Phase 1 specification::

        generate_journeys(request, trains, availability, configuration)

    expressed with explicit keyword arguments and returning the ranked options.
    Use :func:`generate_journeys` when the audit trail is also needed.
    """
    return generate_journeys(
        search_request,
        timetable=timetable,
        availability_book=availability_book,
        configuration=configuration,
        transfer_allowances=transfer_allowances,
        limit=limit,
    ).options


def generate_journeys_for_network(
    search_request: SearchRequest,
    network: object,
    limit: int | None = None,
) -> JourneyGenerationResult:
    """Run the pipeline against a :class:`engine.fixtures.FixtureNetwork`.

    Kept here (rather than in the fixture loader) so that the pipeline owns the
    call signature, and the data-loading layer stays free of engine logic.
    Equivalent to wrapping the network's snapshot in a
    :class:`engine.fixture_provider.FixtureAvailabilityProvider` behind a
    :class:`engine.provider.ProviderAvailabilityBook`, but without the
    per-lookup indirection.
    """
    timetable = getattr(network, "timetable", None)
    availability_book = getattr(network, "availability_book", None)
    configuration = getattr(network, "configuration", None)
    transfer_allowances = getattr(network, "transfer_allowances", ())
    if timetable is None or availability_book is None:
        raise DomainValidationError(
            "generate_journeys_for_network expects an object exposing 'timetable' "
            "and 'availability_book' attributes"
        )
    return generate_journeys(
        search_request,
        timetable=timetable,
        availability_book=availability_book,
        configuration=configuration,
        transfer_allowances=transfer_allowances,
        limit=limit,
    )
