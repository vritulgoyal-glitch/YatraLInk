"""The Phase 1 availability provider: synthetic fixtures behind the contract.

:class:`FixtureAvailabilityProvider` implements the same
:class:`engine.provider.AvailabilityProvider` protocol a future live railway
provider will implement.  Its only responsibility is **data access**: it reads
the synthetic snapshot (via :class:`engine.availability.AvailabilityBook`) and
returns :class:`engine.provider.AvailabilityRecord` values.  It contains no
journey logic — generation, validation and ranking stay in the engine.

The shipped fixtures carry no fetch timestamps, so ``fetched_at`` is ``None``
for this provider; the data contract still transports the field, which is what
a live provider will populate.
"""

from __future__ import annotations

from pathlib import Path

from engine.availability import Availability, AvailabilityBook
from engine.errors import DomainValidationError
from engine.fixtures import DEFAULT_FIXTURE_DIRECTORY, load_availability_book
from engine.provider import AvailabilityQuery, AvailabilityRecord

__all__ = ["FixtureAvailabilityProvider"]


class FixtureAvailabilityProvider:
    """AvailabilityProvider backed by the synthetic fixture snapshot.

    :param book: an already-loaded :class:`AvailabilityBook` (for example
        ``network.availability_book``).  Mutually exclusive with ``directory``.
    :param directory: a fixture directory to load ``availability.json`` from;
        defaults to the shipped snapshot when neither argument is given.
    :param provider_name: provenance identifier reported on every record.
    """

    def __init__(
        self,
        book: AvailabilityBook | None = None,
        *,
        directory: Path | str | None = None,
        provider_name: str = "fixture",
    ) -> None:
        if book is not None and directory is not None:
            raise DomainValidationError(
                "pass either an AvailabilityBook or a fixture directory, not both"
            )
        if book is not None and not isinstance(book, AvailabilityBook):
            raise DomainValidationError(
                f"book must be an AvailabilityBook, got {type(book).__name__}"
            )
        self._book = (
            book
            if book is not None
            else load_availability_book(
                (directory if directory is not None else DEFAULT_FIXTURE_DIRECTORY)
                / "availability.json"
            )
        )
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise DomainValidationError("provider_name must be a non-empty string")
        self._provider_name = provider_name.strip()

    @property
    def provider_name(self) -> str:
        """Provenance identifier of this provider."""
        return self._provider_name

    @property
    def book(self) -> AvailabilityBook:
        """The underlying snapshot, exposed read-only for diagnostics."""
        return self._book

    def get_availability(self, query: AvailabilityQuery) -> AvailabilityRecord | None:
        """Look one bookable unit up in the snapshot.

        Returns ``None`` when the snapshot has no record for the unit; the
        engine renders that as :attr:`AvailabilityState.UNKNOWN`.
        """
        if not isinstance(query, AvailabilityQuery):
            raise DomainValidationError(
                f"query must be an AvailabilityQuery, got {type(query).__name__}"
            )
        record = self._book.lookup(
            train_number=query.train_number,
            origin_station_code=query.origin_station_code,
            destination_station_code=query.destination_station_code,
            travel_date=query.travel_date,
            travel_class=query.travel_class,
            quota=query.quota,
        )
        if record is None:
            return None
        if isinstance(record, Availability):
            travel_date = record.travel_date
            source_updated_at = record.source_updated_at
        else:  # DateAgnosticAvailability: the snapshot declares one state for any date
            travel_date = query.travel_date
            source_updated_at = None
        return AvailabilityRecord(
            train_number=record.train_number,
            origin_station_code=record.origin_station_code,
            destination_station_code=record.destination_station_code,
            travel_date=travel_date,
            travel_class=record.travel_class,
            state=record.state,
            quota=record.quota,
            seats_available=record.seats_available,
            fare=record.fare,
            provider=self._provider_name,
            fetched_at=None,
            updated_at=source_updated_at,
        )
