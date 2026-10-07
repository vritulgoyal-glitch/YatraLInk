"""The availability-provider boundary: contract, fixture provider, engine wiring."""

from __future__ import annotations

from datetime import datetime

import pytest

from engine.enums import AvailabilityState, JourneyType
from engine.errors import DomainValidationError
from engine.fixture_provider import FixtureAvailabilityProvider
from engine.money import Fare
from engine.provider import (
    AvailabilityLookup,
    AvailabilityProvider,
    AvailabilityQuery,
    AvailabilityRecord,
    ProviderAvailabilityBook,
    availability_from_record,
)
from engine.search import SearchRequest
from engine.tests.builders import TRAVEL_DATE, make_train, station

FETCHED_AT = datetime(2026, 6, 1, 6, 0)
UPDATED_AT = datetime(2026, 6, 1, 9, 30)

ALL_STATES = (
    AvailabilityState.AVAILABLE,
    AvailabilityState.RAC,
    AvailabilityState.WAITLIST,
    AvailabilityState.NOT_AVAILABLE,
    AvailabilityState.UNKNOWN,
)


def _record(
    state: AvailabilityState,
    *,
    seats: int | None = 42,
    fare_paise: int | None = 245000,
    provider: str = "unit-test-source",
    fetched_at: datetime | None = FETCHED_AT,
    updated_at: datetime | None = UPDATED_AT,
) -> AvailabilityRecord:
    return AvailabilityRecord(
        train_number="YT1001",
        origin_station_code="BLR",
        destination_station_code="DEL",
        travel_date=TRAVEL_DATE,
        travel_class="3A",
        state=state,
        quota="GENERAL",
        seats_available=seats,
        fare=Fare.from_paise(fare_paise) if fare_paise is not None else None,
        provider=provider,
        fetched_at=fetched_at,
        updated_at=updated_at,
    )


def _provider(
    state: AvailabilityState | None,
    *,
    provider_name: str = "unit-test-source",
    **record_kwargs,
) -> AvailabilityProvider:
    """A minimal hand-written provider implementing the protocol exactly."""

    class ScriptedProvider:
        def __init__(self) -> None:
            self.queries: list[AvailabilityQuery] = []

        @property
        def provider_name(self) -> str:
            return provider_name

        def get_availability(self, query: AvailabilityQuery) -> AvailabilityRecord | None:
            self.queries.append(query)
            if state is None:
                return None
            return _record(state, provider=provider_name, **record_kwargs)

    return ScriptedProvider()


def _book(provider: AvailabilityProvider) -> ProviderAvailabilityBook:
    return ProviderAvailabilityBook(provider)


def _lookup(**overrides):
    kwargs = dict(
        train_number="YT1001",
        origin_station_code="BLR",
        destination_station_code="DEL",
        travel_date=TRAVEL_DATE,
        travel_class="3A",
        quota="GENERAL",
    )
    kwargs.update(overrides)
    return kwargs


class TestAvailabilityQuery:
    def test_normalises_identity_fields(self) -> None:
        query = AvailabilityQuery(
            train_number=" yt1001 ",
            origin_station_code="blr",
            destination_station_code="del",
            travel_date=TRAVEL_DATE,
            travel_class="3A",
            quota=" general ",
        )
        assert query.train_number == "YT1001"
        assert query.origin_station_code == "BLR"
        assert query.quota == "GENERAL"

    def test_rejects_same_origin_and_destination(self) -> None:
        with pytest.raises(DomainValidationError):
            AvailabilityQuery(
                train_number="YT1001",
                origin_station_code="BLR",
                destination_station_code="BLR",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
            )


class TestAvailabilityRecord:
    @pytest.mark.parametrize("state", ALL_STATES, ids=lambda s: s.value)
    def test_each_reported_state_round_trips(self, state: AvailabilityState) -> None:
        record = _record(state)
        assert record.state is state
        assert record.is_confirmed is (state is AvailabilityState.AVAILABLE)

    def test_unknown_is_never_confirmed(self) -> None:
        assert _record(AvailabilityState.UNKNOWN).is_confirmed is False

    def test_seats_available_is_preserved(self) -> None:
        assert _record(AvailabilityState.AVAILABLE, seats=17).seats_available == 17

    def test_fare_is_preserved(self) -> None:
        record = _record(AvailabilityState.AVAILABLE, fare_paise=123456)
        assert record.fare == Fare.from_paise(123456)

    def test_provider_metadata_is_preserved(self) -> None:
        assert _record(AvailabilityState.AVAILABLE, provider="live-rail").provider == "live-rail"

    def test_freshness_metadata_is_preserved(self) -> None:
        record = _record(AvailabilityState.AVAILABLE)
        assert record.fetched_at == FETCHED_AT
        assert record.updated_at == UPDATED_AT

    def test_freshness_metadata_is_optional(self) -> None:
        record = _record(AvailabilityState.AVAILABLE, fetched_at=None, updated_at=None)
        assert record.fetched_at is None
        assert record.updated_at is None

    def test_rejects_negative_seats(self) -> None:
        with pytest.raises(DomainValidationError):
            _record(AvailabilityState.AVAILABLE, seats=-1)

    def test_availability_from_record_keeps_every_reported_fact(self) -> None:
        domain = availability_from_record(_record(AvailabilityState.RAC))
        assert domain.state is AvailabilityState.RAC
        assert domain.seats_available == 42
        assert domain.fare == Fare.from_paise(245000)
        assert domain.source_updated_at == UPDATED_AT

    def test_availability_from_record_falls_back_to_fetched_at(self) -> None:
        domain = availability_from_record(
            _record(AvailabilityState.AVAILABLE, updated_at=None)
        )
        assert domain.source_updated_at == FETCHED_AT

    def test_availability_from_record_rejects_foreign_objects(self) -> None:
        with pytest.raises(DomainValidationError):
            availability_from_record("not a record")  # type: ignore[arg-type]


class TestProviderProtocol:
    def test_hand_written_provider_satisfies_the_protocol(self) -> None:
        provider = _provider(AvailabilityState.AVAILABLE)
        assert isinstance(provider, AvailabilityProvider)

    def test_provider_returning_none_signals_no_data(self) -> None:
        assert _provider(None).get_availability(
            AvailabilityQuery(
                train_number="YT1001",
                origin_station_code="BLR",
                destination_station_code="DEL",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
            )
        ) is None


class TestProviderAvailabilityBook:
    def test_each_state_reaches_the_engine_unchanged(self) -> None:
        for state in ALL_STATES:
            book = _book(_provider(state))
            record = book.lookup(**_lookup())
            assert record is not None
            assert record.state is state

    def test_missing_record_becomes_none(self) -> None:
        assert _book(_provider(None)).lookup(**_lookup()) is None

    def test_seats_fare_and_metadata_survive_the_adapter(self) -> None:
        book = _book(_provider(AvailabilityState.WAITLIST))
        record = book.lookup(**_lookup())
        assert record is not None
        assert record.seats_available == 42
        assert record.fare == Fare.from_paise(245000)
        assert record.source_updated_at == UPDATED_AT

    def test_provider_is_asked_once_per_bookable_unit(self) -> None:
        provider = _provider(AvailabilityState.AVAILABLE)
        book = _book(provider)
        book.lookup(**_lookup())
        book.lookup(**_lookup())
        assert len(provider.queries) == 1

    def test_quota_is_normalised_before_caching(self) -> None:
        provider = _provider(AvailabilityState.AVAILABLE)
        book = _book(provider)
        first = book.lookup(**_lookup(quota="GENERAL"))
        second = book.lookup(**_lookup(quota=" general "))
        assert first is second
        assert len(provider.queries) == 1

    def test_rejects_object_without_get_availability(self) -> None:
        with pytest.raises(DomainValidationError):
            ProviderAvailabilityBook(object())  # type: ignore[arg-type]

    def test_satisfies_the_engine_facing_lookup_protocol(self) -> None:
        assert isinstance(_book(_provider(AvailabilityState.AVAILABLE)), AvailabilityLookup)


class TestFixtureAvailabilityProvider:
    def test_satisfies_the_provider_protocol(self, network) -> None:
        provider = FixtureAvailabilityProvider(network.availability_book)
        assert isinstance(provider, AvailabilityProvider)

    def test_returns_expected_fixture_availability(self) -> None:
        provider = FixtureAvailabilityProvider()
        record = provider.get_availability(
            AvailabilityQuery(
                train_number="YT1001",
                origin_station_code="BLR",
                destination_station_code="DEL",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
            )
        )
        assert record is not None
        assert record.state is AvailabilityState.AVAILABLE
        assert record.seats_available == 42
        assert record.fare == Fare.from_paise(245000)

    def test_reports_each_state_the_snapshot_declares(self, network) -> None:
        provider = FixtureAvailabilityProvider(network.availability_book)
        expected = {
            "YT1001/BLR/DEL": AvailabilityState.AVAILABLE,
            "YT1003/BLR/DEL": AvailabilityState.RAC,
            "YT1001/HYD/DEL": AvailabilityState.WAITLIST,
            "YT1004/HYD/DEL": AvailabilityState.NOT_AVAILABLE,
        }
        for route, state in expected.items():
            number, origin, destination = route.split("/")
            record = provider.get_availability(
                AvailabilityQuery(
                    train_number=number,
                    origin_station_code=origin,
                    destination_station_code=destination,
                    travel_date=TRAVEL_DATE,
                    travel_class="3A",
                )
            )
            assert record is not None, route
            assert record.state is state, route

    def test_unreported_unit_returns_none_not_unknown(self) -> None:
        provider = FixtureAvailabilityProvider()
        record = provider.get_availability(
            AvailabilityQuery(
                train_number="YT1014",
                origin_station_code="HYD",
                destination_station_code="NDLS",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
            )
        )
        assert record is None

    def test_exposes_provider_metadata(self) -> None:
        assert FixtureAvailabilityProvider().provider_name == "fixture"
        custom = FixtureAvailabilityProvider(provider_name="snapshot-2026-06")
        record = custom.get_availability(
            AvailabilityQuery(
                train_number="YT1001",
                origin_station_code="BLR",
                destination_station_code="DEL",
                travel_date=TRAVEL_DATE,
                travel_class="3A",
            )
        )
        assert custom.provider_name == "snapshot-2026-06"
        assert record is not None
        assert record.provider == "snapshot-2026-06"

    def test_rejects_conflicting_construction_arguments(self, network) -> None:
        with pytest.raises(DomainValidationError):
            FixtureAvailabilityProvider(network.availability_book, directory="data/fixtures")


class TestEngineConsumesProvider:
    """The pipeline runs against a provider without knowing its implementation."""

    def test_pipeline_runs_against_a_provider_backed_book(self, network, config) -> None:
        provider = FixtureAvailabilityProvider(network.availability_book)
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = _run_pipeline(request, network.timetable, config, _book(provider))
        assert result.journeys
        assert result.best is not None
        assert result.best.journey_type is JourneyType.DIRECT
        direct = result.best
        assert direct.segments[0].availability is AvailabilityState.AVAILABLE
        assert direct.segments[0].seats_available == 42

    def test_provider_state_flows_into_the_ranked_journey(self, network, config) -> None:
        provider = FixtureAvailabilityProvider(network.availability_book)
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = _run_pipeline(request, network.timetable, config, _book(provider))
        direct = next(
            journey
            for journey in result.options
            if journey.journey_type is JourneyType.DIRECT
            and journey.train_numbers == ("YT1001",)
        )
        assert direct.segments[0].availability is AvailabilityState.AVAILABLE
        assert direct.segments[0].fare == Fare.from_paise(245000)

    def test_engine_output_is_identical_through_provider_or_book(
        self, network, config
    ) -> None:
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        through_book = _run_pipeline(
            request, network.timetable, config, network.availability_book
        )
        through_provider = _run_pipeline(
            request,
            network.timetable,
            config,
            _book(FixtureAvailabilityProvider(network.availability_book)),
        )
        assert through_book.to_json() == through_provider.to_json()

    def test_provider_backed_book_satisfies_the_lookup_protocol(self, network) -> None:
        book = _book(FixtureAvailabilityProvider(network.availability_book))
        assert isinstance(book, AvailabilityLookup)


class TestUnknownNeverConfirmed:
    def test_unknown_provider_record_never_becomes_a_confirmed_segment(self, config) -> None:
        train = make_train("YT9001", [("AAA", None, "06:00", 0), ("BBB", "10:00", None, 0)])
        timetable = _timetable_of(train)
        request = SearchRequest(origin="AAA", destination="BBB", travel_date=TRAVEL_DATE)
        book = _book(_provider(AvailabilityState.UNKNOWN))
        result = _run_pipeline(request, timetable, config, book)
        assert result.journeys
        segment = result.best.segments[0]
        assert segment.availability is AvailabilityState.UNKNOWN
        assert segment.availability.is_confirmed is False

    def test_missing_provider_data_never_becomes_a_confirmed_segment(self, config) -> None:
        train = make_train("YT9001", [("AAA", None, "06:00", 0), ("BBB", "10:00", None, 0)])
        timetable = _timetable_of(train)
        request = SearchRequest(origin="AAA", destination="BBB", travel_date=TRAVEL_DATE)
        book = _book(_provider(None))
        result = _run_pipeline(request, timetable, config, book)
        assert result.journeys
        segment = result.best.segments[0]
        assert segment.availability is AvailabilityState.UNKNOWN
        assert segment.availability.is_confirmed is False
        assert segment.fare is None

    def test_aggregate_of_only_unknown_states_is_never_confirmed(self) -> None:
        from engine.availability import summarize_availability

        summary = summarize_availability(
            [AvailabilityState.UNKNOWN, AvailabilityState.UNKNOWN]
        )
        assert summary.state is AvailabilityState.UNKNOWN
        assert summary.all_confirmed is False
        assert summary.has_unknown is True


def _timetable_of(train):
    from engine.network import Timetable

    return Timetable(stations=(station("AAA"), station("BBB")), trains=(train,))


def _run_pipeline(request, timetable, config, availability):
    from engine.pipeline import generate_journeys

    return generate_journeys(
        request,
        timetable=timetable,
        availability_book=availability,
        configuration=config,
    )
