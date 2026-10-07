"""YatraLink deterministic railway journey engine (Phase 1).

The engine is a pure, deterministic computation over supplied railway data.  It
contains no HTTP framework, no database, no network access and no AI: AI is an
explanation layer that will consume this engine's output in a later phase, never
a source of railway facts.

Subsystem map
-------------

=========================  ==================================================
``engine.models``          domain entities (Station, Train, TrainStop, segments)
``engine.enums``           availability, journey type, risk and rejection enums
``engine.money``           exact money in integer paise
``engine.time_utils``      day-offset aware schedule arithmetic
``engine.network``         timetable snapshot and transfer allowances
``engine.search``          the validated search request
``engine.availability``    reported inventory and its aggregation
``engine.provider``        the availability-provider boundary (protocol, query,
                           record, provider-to-book adapter)
``engine.fixture_provider`` the synthetic-fixture availability provider
``engine.legs``            one ride on one train between two of its stops
``engine.candidates``      direct / same-train split / connecting generation
``engine.connections``     transfer validation and risk classification
``engine.journey``         plan assembly and journey-level validation
``engine.ranking``         explainable, deterministic scoring and ordering
``engine.pipeline``        the end-to-end generate -> validate -> rank pipeline
``engine.fixtures``        loading synthetic fixture snapshots
=========================  ==================================================

Top-level names are resolved lazily (see :pep:`562`) so that importing any single
submodule never triggers a circular import.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "Availability",
    "AvailabilityBook",
    "AvailabilityProvider",
    "AvailabilityQuery",
    "AvailabilityRecord",
    "AvailabilityState",
    "AvailabilitySummary",
    "ConnectionInfo",
    "ConnectionKind",
    "ConnectionRisk",
    "DateAgnosticAvailability",
    "Fare",
    "FixtureAvailabilityProvider",
    "FixtureNetwork",
    "JourneyGenerationResult",
    "JourneyOption",
    "JourneyPlan",
    "JourneyRejection",
    "JourneySegment",
    "JourneyType",
    "ProviderAvailabilityBook",
    "RailLeg",
    "RankedJourney",
    "RankingWeights",
    "RejectionReason",
    "SearchConfiguration",
    "SearchRequest",
    "Station",
    "Timetable",
    "Train",
    "TrainStop",
    "TravelClass",
    "TransferAllowance",
    "assess_connection",
    "default_network",
    "generate_journey_options",
    "generate_journeys",
    "load_fixture_network",
    "rank_journeys",
    "score_journey",
    "sort_journeys",
    "summarize_availability",
    "validate_connection",
    "__version__",
]

_LAZY_IMPORTS: dict[str, str] = {
    "Availability": "engine.availability",
    "AvailabilityBook": "engine.availability",
    "AvailabilitySummary": "engine.availability",
    "AvailabilityProvider": "engine.provider",
    "AvailabilityQuery": "engine.provider",
    "AvailabilityRecord": "engine.provider",
    "ProviderAvailabilityBook": "engine.provider",
    "FixtureAvailabilityProvider": "engine.fixture_provider",
    "DateAgnosticAvailability": "engine.availability",
    "summarize_availability": "engine.availability",
    "generate_journey_options": "engine.pipeline",
    "generate_journeys": "engine.pipeline",
    "JourneyGenerationResult": "engine.pipeline",
    "JourneyRejection": "engine.pipeline",
    "JourneyPlan": "engine.candidates",
    "assess_connection": "engine.connections",
    "validate_connection": "engine.connections",
    "RankingWeights": "engine.config",
    "SearchConfiguration": "engine.config",
    "AvailabilityState": "engine.enums",
    "ConnectionKind": "engine.enums",
    "ConnectionRisk": "engine.enums",
    "JourneyType": "engine.enums",
    "RejectionReason": "engine.enums",
    "TravelClass": "engine.enums",
    "FixtureNetwork": "engine.fixtures",
    "default_network": "engine.fixtures",
    "load_fixture_network": "engine.fixtures",
    "RailLeg": "engine.legs",
    "ConnectionInfo": "engine.models",
    "JourneyOption": "engine.models",
    "JourneySegment": "engine.models",
    "Station": "engine.models",
    "Train": "engine.models",
    "TrainStop": "engine.models",
    "Fare": "engine.money",
    "Timetable": "engine.network",
    "TransferAllowance": "engine.network",
    "rank_journeys": "engine.ranking",
    "RankedJourney": "engine.ranking",
    "score_journey": "engine.ranking",
    "sort_journeys": "engine.ranking",
    "SearchRequest": "engine.search",
}


def __getattr__(name: str) -> Any:
    """Import a documented top-level name on first access."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
