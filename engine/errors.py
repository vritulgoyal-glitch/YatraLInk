"""Domain error types raised by the YatraLink engine.

The engine never swallows an error silently.  Invalid *input data* and invalid
*candidate journeys* raise these exceptions; the journey pipeline catches the
journey-level error only to record an explicit rejection reason (see
:class:`engine.pipeline.JourneyRejection`) and never to hide a defect.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CandidateLimitError",
    "DataSourceError",
    "DomainValidationError",
    "InvalidJourneyError",
    "SearchRequestError",
    "UnknownStationError",
    "YatraLinkEngineError",
]


class YatraLinkEngineError(Exception):
    """Base class for every error raised by the engine."""


class DomainValidationError(YatraLinkEngineError, ValueError):
    """Supplied railway/domain data violates a documented invariant.

    Also a :class:`ValueError` so callers can treat engine validation like any
    other Python validation error.
    """


class SearchRequestError(DomainValidationError):
    """The search request is malformed or internally inconsistent."""


class InvalidJourneyError(DomainValidationError):
    """A candidate journey failed validation and must not be ranked.

    :param reason: machine-readable :class:`engine.enums.RejectionReason` when
        the failure comes from candidate validation.
    """

    def __init__(self, message: str, *, reason: Any | None = None) -> None:
        super().__init__(message)
        self.reason = reason


class UnknownStationError(DomainValidationError):
    """A station code is not part of the route being inspected."""


class DataSourceError(YatraLinkEngineError):
    """Railway data could not be loaded from a data source."""


class CandidateLimitError(YatraLinkEngineError):
    """Candidate generation exceeded a configured safety limit."""
