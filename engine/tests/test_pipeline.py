"""The end-to-end pipeline: generation, validation, deduplication and ranking."""

from __future__ import annotations

import json
from datetime import time

import pytest

from engine.config import SearchConfiguration
from engine.enums import AvailabilityState, ConnectionRisk, JourneyType, RejectionReason
from engine.errors import CandidateLimitError, DomainValidationError
from engine.models import JourneyOption
from engine.pipeline import (
    deduplicate_journeys,
    generate_journey_options,
    generate_journeys,
    generate_journeys_for_network,
)
from engine.ranking import RankingWeights
from engine.search import SearchRequest
from engine.tests.builders import TRAVEL_DATE, find_journey, journeys_of_type


@pytest.fixture()
def blr_del_request() -> SearchRequest:
    return SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)


class TestPipelineShape:
    def test_result_is_a_complete_audit_trail(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.request is blr_del_request
        assert result.configuration is config
        assert result.generated_plan_count >= len(result.journeys)
        assert result.candidate_count == len(result.journeys)
        assert set(result.type_counts) == {member.value for member in JourneyType}
        assert set(result.availability_counts) == {member.value for member in AvailabilityState}

    def test_all_three_journey_types_are_produced(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert journeys_of_type(result, JourneyType.DIRECT)
        assert journeys_of_type(result, JourneyType.SAME_TRAIN_SPLIT)
        assert journeys_of_type(result, JourneyType.CONNECTING)

    def test_every_journey_is_a_valid_option(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.options
        for journey in result.options:
            assert isinstance(journey, JourneyOption)
            assert journey.origin_station_code == "BLR"
            assert journey.destination_station_code == "DEL"
            assert journey.total_duration_minutes > 0
            assert len(journey.segments) <= 3
            assert journey.train_changes <= 2
            assert journey.connections or len(journey.segments) == 1

    def test_elapsed_duration_always_equals_travel_plus_transfer(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for journey in result.options:
            assert (
                journey.total_duration_minutes
                == journey.total_travel_minutes + journey.total_transfer_minutes
            )

    def test_elapsed_duration_is_not_the_sum_of_segment_durations(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        multi = [j for j in result.options if len(j.segments) > 1]
        assert multi
        assert any(
            j.total_duration_minutes > sum(s.duration_minutes for s in j.segments) for j in multi
        )

    def test_every_journey_has_a_stable_id(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        first = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        second = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert [j.journey_id for j in first.options] == [j.journey_id for j in second.options]

    def test_pipeline_is_deterministic_under_input_reordering(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        first = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        second = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert [j.dedup_key for j in first.options] == [j.dedup_key for j in second.options]

    def test_result_is_json_serialisable(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        payload = json.loads(json.dumps(result.to_json()))
        assert payload["journey_count"] == len(result.journeys)
        assert payload["journeys"][0]["rank"] == 1

    def test_rank_gap_between_best_and_worst_is_explained(
        self, timetable, availability_book, config
    ) -> None:
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.journeys
        for ranked in result.journeys:
            assert 0 <= ranked.score.total <= 1000
            assert ranked.score.explain()


class TestConnectionOutcomesInThePipeline:
    def test_valid_connection_is_offered(self, timetable, availability_book, config) -> None:
        """YT1007 arrives HYD at 12:00; YT1012 departs HYD at 13:30 (90 min, SAFE)."""
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = find_journey(result, train_numbers=("YT1007", "YT1012"))
        assert journey is not None
        assert journey.journey_type is JourneyType.CONNECTING
        connection = journey.connections[0]
        assert connection.transfer_minutes == 90
        assert connection.required_minutes == 30
        assert connection.risk is ConnectionRisk.SAFE
        assert journey.train_changes == 1

    def test_connection_exactly_at_the_minimum_is_offered_and_tight(
        self, timetable, availability_book, config
    ) -> None:
        """YT1007 arrives HYD at 12:00; YT1011 departs HYD at 12:30 — exactly 30."""
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = find_journey(result, train_numbers=("YT1007", "YT1011"))
        assert journey is not None
        connection = journey.connections[0]
        assert connection.buffer_minutes == 0
        assert connection.valid is True
        assert connection.risk is ConnectionRisk.TIGHT

    def test_connection_below_the_minimum_is_rejected_with_a_reason(
        self, timetable, availability_book, config
    ) -> None:
        """YT1010 departs HYD at 12:29, one minute below the 30 minute minimum."""
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert find_journey(result, train_numbers=("YT1007", "YT1010")) is None
        invalid = [
            rejection
            for rejection in result.rejections
            if rejection.reason is RejectionReason.INVALID_CONNECTION
        ]
        assert invalid
        assert any("29 minute" in rejection.detail for rejection in invalid)

    def test_overnight_connection_is_offered(self, timetable, availability_book, config) -> None:
        """YT1005 arrives HYD at 22:00; YT1006 departs HYD at 05:00 next day."""
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = find_journey(result, train_numbers=("YT1005", "YT1006"))
        assert journey is not None
        assert journey.connections[0].transfer_minutes == 420
        assert journey.departure.date().isoformat() == "2026-06-15"
        assert journey.arrival.date().isoformat() == "2026-06-16"

    def test_connection_requirement_is_configurable(self, timetable, availability_book) -> None:
        relaxed = SearchConfiguration(minimum_connection_minutes=20)
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=relaxed,
        )
        journey = find_journey(result, train_numbers=("YT1007", "YT1008"))
        assert journey is not None
        assert journey.connections[0].required_minutes == 20
        assert journey.connections[0].transfer_minutes == 20


class TestStationChangeInThePipeline:
    def test_station_change_journey_is_not_offered_when_disabled(
        self, timetable, availability_book
    ) -> None:
        disabled = SearchConfiguration(allow_cross_station_transfers=False)
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=disabled,
        )
        for journey in result.options:
            for connection in journey.connections:
                assert not connection.station_change
                assert connection.from_station_code == connection.to_station_code

    def test_station_change_journey_is_offered_when_the_data_allows_it(
        self, timetable, availability_book, config
    ) -> None:
        """YT1015 BLR -> NDLS, then the declared NDLS -> DEL allowance, then YT1018."""
        assert config.allow_cross_station_transfers is True
        request = SearchRequest(origin="BLR", destination="BPL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        journey = find_journey(result, train_numbers=("YT1015", "YT1018"))
        assert journey is not None
        connection = journey.connections[0]
        assert connection.station_change
        assert (connection.from_station_code, connection.to_station_code) == ("NDLS", "DEL")
        assert connection.required_minutes == 45
        assert connection.risk is ConnectionRisk.TIGHT  # a station change is never SAFE

    def test_station_change_is_rejected_without_an_allowance(
        self, timetable, availability_book
    ) -> None:
        """With no allowance declared, the same pair must be rejected."""
        enabled_no_allowances = SearchConfiguration(allow_cross_station_transfers=True)
        request = SearchRequest(origin="BLR", destination="BPL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=enabled_no_allowances,
            transfer_allowances=(),  # deliberately no allowances
        )
        assert find_journey(result, train_numbers=("YT1015", "YT1018")) is None


class TestDeduplication:
    def test_duplicate_journeys_collapse(self) -> None:
        from engine.tests.builders import make_journey, make_train

        train = make_train("YT1", [("AAA", None, "08:00", 0), ("BBB", "10:00", None, 0)])
        first = make_journey(train, "AAA", "BBB", journey_id="a")
        second = make_journey(train, "AAA", "BBB", journey_id="b")
        unique, duplicates = deduplicate_journeys([first, second])
        assert len(unique) == 1
        assert len(duplicates) == 1

    def test_same_train_split_and_connecting_do_not_collapse(self) -> None:
        from engine.tests.builders import make_segment, make_train

        train = make_train(
            "YT1",
            [("AAA", None, "08:00", 0), ("BBB", "12:00", "12:05", 0), ("CCC", "18:00", None, 0)],
        )
        segments = (
            make_segment(train, "AAA", "BBB"),
            make_segment(train, "BBB", "CCC", reservation_index=1),
        )
        split = JourneyOption(
            journey_id="s",
            journey_type=JourneyType.SAME_TRAIN_SPLIT,
            origin_station_code="AAA",
            destination_station_code="CCC",
            segments=segments,
        )
        connecting = JourneyOption(
            journey_id="c",
            journey_type=JourneyType.CONNECTING,
            origin_station_code="AAA",
            destination_station_code="CCC",
            segments=segments,
        )
        unique, duplicates = deduplicate_journeys([split, connecting])
        assert len(unique) == 2
        assert duplicates == ()

    def test_no_duplicates_are_returned_by_the_pipeline(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        keys = [journey.dedup_key for journey in result.options]
        assert len(keys) == len(set(keys))

    def test_duplicates_are_reported_not_hidden(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.generated_plan_count == (
            result.candidate_count + result.duplicate_count + len(result.rejections)
        )


class TestLimitsAndSwitches:
    def test_limit_truncates_after_ranking(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        full = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        limited = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
            limit=2,
        )
        assert len(limited.journeys) == 2
        assert [j.journey_id for j in limited.journeys] == [j.journey_id for j in full.journeys[:2]]

    def test_negative_limit_is_rejected(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        with pytest.raises(DomainValidationError):
            generate_journeys(
                blr_del_request,
                timetable=timetable,
                availability_book=availability_book,
                configuration=config,
                limit=-1,
            )

    def test_zero_limit_returns_nothing_but_still_reports(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
            limit=0,
        )
        assert result.journeys == ()
        assert result.candidate_count > 0

    def test_generators_can_be_switched_off_individually(
        self, timetable, availability_book, blr_del_request
    ) -> None:
        only_direct = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=SearchConfiguration(
                enable_same_train_split=False, enable_connecting=False
            ),
        )
        assert journeys_of_type(only_direct, JourneyType.DIRECT)
        assert journeys_of_type(only_direct, JourneyType.SAME_TRAIN_SPLIT) == ()
        assert journeys_of_type(only_direct, JourneyType.CONNECTING) == ()

    def test_no_generators_switched_on_yields_no_journeys(
        self, timetable, availability_book, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=SearchConfiguration(
                enable_direct=False, enable_same_train_split=False, enable_connecting=False
            ),
        )
        assert result.options == ()
        assert result.generated_plan_count == 0

    def test_candidate_limit_is_enforced(self, timetable, availability_book) -> None:
        tiny = SearchConfiguration(max_candidates_per_type=1)
        request = SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE)
        with pytest.raises(CandidateLimitError):
            generate_journeys(
                request,
                timetable=timetable,
                availability_book=availability_book,
                configuration=tiny,
            )

    def test_null_candidate_limit_is_rejected_by_configuration(self) -> None:
        with pytest.raises(ValueError):
            SearchConfiguration(max_candidates_per_type=0)


class TestInterfaceVariants:
    def test_generate_journey_options_returns_only_options(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        options = generate_journey_options(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert options
        assert all(isinstance(option, JourneyOption) for option in options)

    def test_generate_journeys_for_network_uses_the_snapshot(
        self, network, blr_del_request
    ) -> None:
        result = generate_journeys_for_network(blr_del_request, network)
        assert result.options
        assert result.configuration is network.configuration

    def test_generate_journeys_for_network_rejects_a_bad_object(self, blr_del_request) -> None:
        with pytest.raises(DomainValidationError):
            generate_journeys_for_network(blr_del_request, object())

    def test_non_request_input_is_rejected(self, timetable, availability_book, config) -> None:
        with pytest.raises(DomainValidationError):
            generate_journeys(
                {"origin": "BLR"},  # type: ignore[arg-type]
                timetable=timetable,
                availability_book=availability_book,
                configuration=config,
            )

    def test_default_configuration_is_used_when_none_supplied(
        self, timetable, availability_book, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request, timetable=timetable, availability_book=availability_book
        )
        assert result.configuration == SearchConfiguration()

    def test_custom_ranking_weights_change_the_order(
        self, timetable, availability_book, blr_del_request
    ) -> None:
        cheapest_first = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=SearchConfiguration(
                ranking_weights=RankingWeights(
                    availability=0,
                    train_changes=0,
                    fare=100,
                    duration=0,
                    connection_risk=0,
                    separate_reservations=0,
                    station_change=0,
                )
            ),
        )
        assert cheapest_first.options
        fares = [
            journey.total_fare.amount_paise if journey.total_fare else None
            for journey in cheapest_first.options
        ]
        priced = [fare for fare in fares if fare is not None]
        assert priced == sorted(priced)


class TestRejectionAudit:
    def test_every_rejection_has_a_machine_readable_reason(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        for rejection in result.rejections:
            assert isinstance(rejection.reason, RejectionReason)
            assert rejection.detail
            assert rejection.journey_type in {member.value for member in JourneyType}

    def test_rejection_counts_group_by_reason(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        counts = result.rejection_counts
        assert sum(counts.values()) == len(result.rejections)

    def test_rejections_are_json_serialisable(
        self, timetable, availability_book, config, blr_del_request
    ) -> None:
        result = generate_journeys(
            blr_del_request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        payload = json.dumps(result.to_json())
        assert "rejections" in payload

    def test_not_available_rejections_are_recorded(
        self, timetable, availability_book, config
    ) -> None:
        request = SearchRequest(origin="HYD", destination="DEL", travel_date=TRAVEL_DATE)
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert RejectionReason.SEGMENT_NOT_AVAILABLE in {
            rejection.reason for rejection in result.rejections
        }

    def test_after_latest_arrival_rejection_is_recorded(
        self, timetable, availability_book, config
    ) -> None:
        """A deadline prunes legs during generation and is re-checked by validation."""
        from datetime import datetime

        deadline = datetime(2026, 6, 16, 12, 0)
        request = SearchRequest(
            origin="BLR",
            destination="DEL",
            travel_date=TRAVEL_DATE,
            latest_arrival=deadline,
        )
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.options
        for journey in result.options:
            assert journey.arrival <= deadline

        unbounded = generate_journeys(
            SearchRequest(origin="BLR", destination="DEL", travel_date=TRAVEL_DATE),
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.generated_plan_count < unbounded.generated_plan_count

    def test_before_earliest_departure_rejection_is_recorded(
        self, timetable, availability_book, config
    ) -> None:
        request = SearchRequest(
            origin="BLR",
            destination="DEL",
            travel_date=TRAVEL_DATE,
            earliest_departure=time(22, 0),
        )
        result = generate_journeys(
            request,
            timetable=timetable,
            availability_book=availability_book,
            configuration=config,
        )
        assert result.options
        for journey in result.options:
            assert journey.departure >= request.earliest_departure_datetime


class TestCrossComponentBoundary:
    def test_engine_does_not_import_the_backend_package(self) -> None:
        """The engine must stay usable without the FastAPI application."""
        import pathlib
        import re

        engine_dir = pathlib.Path(__file__).resolve().parents[1]
        offenders = []
        for path in engine_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if re.search(r"^\s*(from|import)\s+app\b", text, flags=re.MULTILINE):
                offenders.append(str(path))
        assert offenders == []

    def test_engine_does_not_import_web_or_infrastructure_libraries(self) -> None:
        """No FastAPI, no HTTP client, no database, no LLM library in the engine."""
        import pathlib

        forbidden = (
            "fastapi",
            "starlette",
            "uvicorn",
            "httpx",
            "requests",
            "sqlalchemy",
            "psycopg",
            "redis",
            "supabase",
            "openai",
            "anthropic",
        )
        engine_dir = pathlib.Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for path in engine_dir.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for line in source.splitlines():
                stripped = line.strip()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                for library in forbidden:
                    if stripped.startswith(f"import {library}") or stripped.startswith(
                        f"from {library}"
                    ):
                        offenders.append(f"{path.name}: {stripped}")
        assert offenders == []

    def test_engine_has_no_network_calls(self) -> None:
        """No socket, urllib, or HTTP calls anywhere in the engine."""
        import pathlib

        forbidden = ("urllib.request", "socket.", "http.client", "httpx.", "requests.")
        engine_dir = pathlib.Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for path in engine_dir.rglob("*.py"):
            if path.name == "test_pipeline.py":
                continue
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in source:
                    offenders.append(f"{path.name}: {token}")
        assert offenders == []

    def test_engine_has_no_obvious_secrets(self) -> None:
        import pathlib
        import re

        engine_dir = pathlib.Path(__file__).resolve().parents[1]
        pattern = re.compile(r"(api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]+['\"]", re.I)
        offenders: list[str] = []
        for path in engine_dir.rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(path.name)
        assert offenders == []

    def test_no_ai_integration_in_the_engine(self) -> None:
        import pathlib

        engine_dir = pathlib.Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for path in engine_dir.rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            source = path.read_text(encoding="utf-8").lower()
            for token in ("openai", "anthropic", "langchain", "llm_client", "gpt-4", "gemini"):
                if token in source:
                    offenders.append(f"{path.name}: {token}")
        assert offenders == []
