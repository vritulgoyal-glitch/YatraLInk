"""Ranking: explainable scoring, determinism and documented tiebreakers."""

from __future__ import annotations

import random

import pytest

from engine.config import RankingWeights, SearchConfiguration
from engine.enums import (
    AvailabilityState,
    ConnectionKind,
    ConnectionRisk,
    JourneyType,
)
from engine.models import ConnectionInfo, JourneyOption
from engine.ranking import (
    MAX_SUBSCORE,
    ScoreContext,
    ScoreScale,
    build_score_context,
    compare_journeys,
    normalise,
    rank_journeys,
    score_journey,
    sort_journeys,
)
from engine.tests.builders import make_journey, make_segment, make_train

WEIGHTS = RankingWeights()


@pytest.fixture()
def direct_train():
    return make_train("YT9001", [("AAA", None, "08:00", 0), ("CCC", "18:00", None, 0)])


@pytest.fixture()
def split_trains():
    """Two trains whose combined timetable reproduces the direct journey."""
    first = make_train("YT9002", [("AAA", None, "08:00", 0), ("BBB", "12:00", None, 0)])
    second = make_train("YT9003", [("BBB", None, "14:00", 0), ("CCC", "18:00", None, 0)])
    return first, second


class TestScoreScale:
    def test_every_dimension_is_scored(self, direct_train) -> None:
        journey = make_journey(direct_train, "AAA", "CCC")
        context = build_score_context([journey])
        score = score_journey(journey, context=context, weights=WEIGHTS)
        assert set(score.as_dict) == set(ScoreScale.ORDER)

    def test_total_is_within_scale(self, direct_train) -> None:
        journey = make_journey(direct_train, "AAA", "CCC")
        score = score_journey(journey, context=build_score_context([journey]), weights=WEIGHTS)
        assert 0 <= score.total <= MAX_SUBSCORE

    def test_perfect_single_candidate_scores_the_maximum(self, direct_train) -> None:
        """A lone fully-available direct journey is the best of its own set."""
        journey = make_journey(direct_train, "AAA", "CCC")
        score = score_journey(journey, context=build_score_context([journey]), weights=WEIGHTS)
        assert score.total == MAX_SUBSCORE

    def test_contributions_match_weights_times_subscores(self, direct_train) -> None:
        journey = make_journey(direct_train, "AAA", "CCC")
        score = score_journey(journey, context=build_score_context([journey]), weights=WEIGHTS)
        for component in score.components:
            assert component.contribution == component.weight * component.subscore

    def test_explanation_is_human_readable(self, direct_train) -> None:
        journey = make_journey(direct_train, "AAA", "CCC")
        score = score_journey(journey, context=build_score_context([journey]), weights=WEIGHTS)
        text = score.explain()
        assert "total=" in text
        assert ScoreScale.AVAILABILITY in text

    def test_component_lookup(self, direct_train) -> None:
        journey = make_journey(direct_train, "AAA", "CCC")
        score = score_journey(journey, context=build_score_context([journey]), weights=WEIGHTS)
        assert score.component(ScoreScale.FARE).subscore == MAX_SUBSCORE
        with pytest.raises(KeyError):
            score.component("nonexistent")

    def test_to_json_is_serialisable(self, direct_train) -> None:
        import json

        journey = make_journey(direct_train, "AAA", "CCC")
        score = score_journey(journey, context=build_score_context([journey]), weights=WEIGHTS)
        assert json.loads(json.dumps(score.to_json()))["total"] == score.total

    def test_no_floats_anywhere_in_the_score(self, direct_train) -> None:
        journey = make_journey(direct_train, "AAA", "CCC")
        score = score_journey(journey, context=build_score_context([journey]), weights=WEIGHTS)
        assert isinstance(score.total, int)
        for component in score.components:
            assert isinstance(component.subscore, int)
            assert isinstance(component.contribution, int)


class TestNormalise:
    def test_monotonic_for_lower_is_better(self) -> None:
        values = [
            normalise(v, best=0, worst=100, higher_is_better=False) for v in range(0, 101, 10)
        ]
        assert values == sorted(values, reverse=True)

    def test_monotonic_for_higher_is_better(self) -> None:
        # For a higher-is-better dimension the "best" reference is the largest value.
        values = [normalise(v, best=100, worst=0, higher_is_better=True) for v in range(0, 101, 10)]
        assert values == sorted(values)

    def test_deterministic(self) -> None:
        first = [normalise(v, best=1, worst=97, higher_is_better=False) for v in range(1, 98, 7)]
        second = [normalise(v, best=1, worst=97, higher_is_better=False) for v in range(1, 98, 7)]
        assert first == second


class TestPreferenceOrdering:
    def test_better_availability_ranks_first(self, direct_train) -> None:
        good = make_journey(
            direct_train, "AAA", "CCC", journey_id="good", availability=AvailabilityState.AVAILABLE
        )
        weak = make_journey(
            direct_train, "AAA", "CCC", journey_id="weak", availability=AvailabilityState.RAC
        )
        ordered = sort_journeys([weak, good], weights=WEIGHTS)
        assert ordered[0].journey_id == "good"
        assert ordered[0].rank == 1
        assert ordered[1].journey_id == "weak"

    def test_waitlist_ranks_below_rac(self, direct_train) -> None:
        rac = make_journey(
            direct_train, "AAA", "CCC", journey_id="rac", availability=AvailabilityState.RAC
        )
        waitlist = make_journey(
            direct_train, "AAA", "CCC", journey_id="wl", availability=AvailabilityState.WAITLIST
        )
        ordered = sort_journeys([waitlist, rac], weights=WEIGHTS)
        assert [r.journey_id for r in ordered] == ["rac", "wl"]

    def test_unknown_ranks_below_waitlist(self, direct_train) -> None:
        waitlist = make_journey(
            direct_train, "AAA", "CCC", journey_id="wl", availability=AvailabilityState.WAITLIST
        )
        unknown = make_journey(
            direct_train, "AAA", "CCC", journey_id="unknown", availability=AvailabilityState.UNKNOWN
        )
        ordered = sort_journeys([unknown, waitlist], weights=WEIGHTS)
        assert [r.journey_id for r in ordered] == ["wl", "unknown"]

    def test_unknown_never_scores_as_confirmed(self, direct_train) -> None:
        confirmed = make_journey(
            direct_train, "AAA", "CCC", journey_id="c", availability=AvailabilityState.AVAILABLE
        )
        unknown = make_journey(
            direct_train, "AAA", "CCC", journey_id="u", availability=AvailabilityState.UNKNOWN
        )
        context = build_score_context([confirmed, unknown])
        confirmed_score = score_journey(confirmed, context=context, weights=WEIGHTS)
        unknown_score = score_journey(unknown, context=context, weights=WEIGHTS)
        assert (
            confirmed_score.component(ScoreScale.AVAILABILITY).subscore
            > unknown_score.component(ScoreScale.AVAILABILITY).subscore
        )

    def test_lower_fare_ranks_first(self, direct_train) -> None:
        cheap = make_journey(direct_train, "AAA", "CCC", journey_id="cheap", fare_paise=100000)
        dear = make_journey(direct_train, "AAA", "CCC", journey_id="dear", fare_paise=200000)
        ordered = sort_journeys([dear, cheap], weights=WEIGHTS)
        assert [r.journey_id for r in ordered] == ["cheap", "dear"]

    def test_shorter_duration_ranks_first(self) -> None:
        fast = make_journey(
            make_train("YT9001", [("AAA", None, "08:00", 0), ("CCC", "14:00", None, 0)]),
            "AAA",
            "CCC",
            journey_id="fast",
        )
        slow = make_journey(
            make_train("YT9002", [("AAA", None, "08:00", 0), ("CCC", "20:00", None, 0)]),
            "AAA",
            "CCC",
            journey_id="slow",
        )
        ordered = sort_journeys([slow, fast], weights=WEIGHTS)
        assert [r.journey_id for r in ordered] == ["fast", "slow"]

    def test_unknown_fare_never_outranks_a_priced_journey(self, direct_train) -> None:
        priced = make_journey(direct_train, "AAA", "CCC", journey_id="priced", fare_paise=500000)
        unpriced = make_journey(direct_train, "AAA", "CCC", journey_id="unpriced", fare_paise=None)
        context = build_score_context([priced, unpriced])
        priced_score = score_journey(priced, context=context, weights=WEIGHTS)
        unpriced_score = score_journey(unpriced, context=context, weights=WEIGHTS)
        assert unpriced_score.component(ScoreScale.FARE).subscore == 0
        assert priced_score.total > unpriced_score.total

    def test_fewer_train_changes_are_preferred(self, direct_train, split_trains) -> None:
        first, second = split_trains
        direct = make_journey(direct_train, "AAA", "CCC", journey_id="direct")
        direct = JourneyOption(
            journey_id="direct",
            journey_type=JourneyType.DIRECT,
            origin_station_code="AAA",
            destination_station_code="CCC",
            segments=(make_segment(direct_train, "AAA", "CCC", fare_paise=200000),),
        )
        connecting = JourneyOption(
            journey_id="connecting",
            journey_type=JourneyType.CONNECTING,
            origin_station_code="AAA",
            destination_station_code="CCC",
            segments=(
                make_segment(first, "AAA", "BBB", fare_paise=100000),
                make_segment(second, "BBB", "CCC", reservation_index=1, fare_paise=100000),
            ),
        )
        assert direct.total_duration_minutes == connecting.total_duration_minutes
        assert direct.total_fare == connecting.total_fare

        context = build_score_context([direct, connecting])
        direct_score = score_journey(direct, context=context, weights=WEIGHTS)
        connecting_score = score_journey(connecting, context=context, weights=WEIGHTS)
        assert direct_score.component(ScoreScale.TRAIN_CHANGES).subscore == MAX_SUBSCORE
        assert connecting_score.component(ScoreScale.TRAIN_CHANGES).subscore == 0
        ordered = sort_journeys([connecting, direct], weights=WEIGHTS)
        assert [r.journey_id for r in ordered] == ["direct", "connecting"]

    def test_safer_connection_scores_higher(self, split_trains) -> None:
        first, second = split_trains
        base_segments = (
            make_segment(first, "AAA", "BBB", fare_paise=100000),
            make_segment(second, "BBB", "CCC", reservation_index=1, fare_paise=100000),
        )

        def with_risk(risk: ConnectionRisk, journey_id: str) -> JourneyOption:
            safe_segments = base_segments
            arrival = safe_segments[0].arrival
            departure = safe_segments[1].departure
            connection = ConnectionInfo(
                index=0,
                kind=ConnectionKind.CROSS_TRAIN_SAME_STATION,
                arrival=arrival,
                departure=departure,
                transfer_minutes=120,
                required_minutes=30,
                valid=True,
                risk=risk,
                from_station_code="BBB",
                to_station_code="BBB",
                buffer_minutes=90 if risk is ConnectionRisk.SAFE else 0,
            )
            return JourneyOption(
                journey_id=journey_id,
                journey_type=JourneyType.CONNECTING,
                origin_station_code="AAA",
                destination_station_code="CCC",
                segments=safe_segments,
                connections=(connection,),
            )

        safe = with_risk(ConnectionRisk.SAFE, "safe")
        tight = with_risk(ConnectionRisk.TIGHT, "tight")

        # A context where only the risk dimension can differ, so the comparison
        # isolates the risk score instead of being confounded by price or time.
        context = ScoreContext(
            min_duration_minutes=safe.total_duration_minutes,
            max_duration_minutes=safe.total_duration_minutes,
            min_fare_paise=200000,
            max_fare_paise=200000,
            min_train_changes=1,
            max_train_changes=1,
            min_reservations=2,
            max_reservations=2,
            min_station_changes=0,
            max_station_changes=0,
        )
        safe_score = score_journey(safe, context=context, weights=WEIGHTS)
        tight_score = score_journey(tight, context=context, weights=WEIGHTS)
        assert safe_score.component(ScoreScale.CONNECTION_RISK).subscore == MAX_SUBSCORE
        assert tight_score.component(ScoreScale.CONNECTION_RISK).subscore == MAX_SUBSCORE // 2
        assert safe_score.total > tight_score.total


class TestTiebreakers:
    def test_direct_is_preferred_over_same_train_split(self) -> None:
        train = make_train(
            "YT9001",
            [("AAA", None, "08:00", 0), ("BBB", "12:00", "12:05", 0), ("CCC", "18:00", None, 0)],
        )
        direct = JourneyOption(
            journey_id="d",
            journey_type=JourneyType.DIRECT,
            origin_station_code="AAA",
            destination_station_code="CCC",
            segments=(make_segment(train, "AAA", "CCC"),),
        )
        split = JourneyOption(
            journey_id="s",
            journey_type=JourneyType.SAME_TRAIN_SPLIT,
            origin_station_code="AAA",
            destination_station_code="CCC",
            segments=(
                make_segment(train, "AAA", "BBB"),
                make_segment(train, "BBB", "CCC", reservation_index=1),
            ),
        )
        assert compare_journeys(direct, split) < 0
        assert compare_journeys(split, direct) > 0

    def test_same_train_split_is_preferred_over_connecting(self, split_trains) -> None:
        first, second = split_trains
        split_train = make_train(
            "YT9004",
            [("AAA", None, "08:00", 0), ("BBB", "12:00", "12:05", 0), ("CCC", "18:00", None, 0)],
        )
        split = JourneyOption(
            journey_id="s",
            journey_type=JourneyType.SAME_TRAIN_SPLIT,
            origin_station_code="AAA",
            destination_station_code="CCC",
            segments=(
                make_segment(split_train, "AAA", "BBB"),
                make_segment(split_train, "BBB", "CCC", reservation_index=1),
            ),
        )
        connecting = JourneyOption(
            journey_id="c",
            journey_type=JourneyType.CONNECTING,
            origin_station_code="AAA",
            destination_station_code="CCC",
            segments=(
                make_segment(first, "AAA", "BBB"),
                make_segment(second, "BBB", "CCC", reservation_index=1),
            ),
        )
        assert compare_journeys(split, connecting) < 0

    def test_identical_journeys_compare_equal(self, direct_train) -> None:
        first = make_journey(direct_train, "AAA", "CCC", journey_id="same")
        second = make_journey(direct_train, "AAA", "CCC", journey_id="same")
        assert compare_journeys(first, second) == 0


class TestDeterminism:
    def test_ranking_is_repeatable(self, direct_train) -> None:
        journeys = [
            make_journey(
                direct_train,
                "AAA",
                "CCC",
                journey_id=f"j{index}",
                availability=state,
                fare_paise=100000 + index * 10000,
            )
            for index, state in enumerate(
                [
                    AvailabilityState.AVAILABLE,
                    AvailabilityState.RAC,
                    AvailabilityState.WAITLIST,
                    AvailabilityState.UNKNOWN,
                    AvailabilityState.AVAILABLE,
                ]
            )
        ]
        first = [r.journey_id for r in sort_journeys(journeys, weights=WEIGHTS)]
        second = [r.journey_id for r in sort_journeys(journeys, weights=WEIGHTS)]
        assert first == second

    def test_input_order_does_not_change_the_result(self, direct_train) -> None:
        journeys = [
            make_journey(
                direct_train,
                "AAA",
                "CCC",
                journey_id=f"j{index}",
                fare_paise=100000 + index * 25000,
            )
            for index in range(6)
        ]
        reference = [r.journey_id for r in sort_journeys(journeys, weights=WEIGHTS)]
        for seed in range(5):
            shuffled = list(journeys)
            random.Random(seed).shuffle(shuffled)
            assert [r.journey_id for r in sort_journeys(shuffled, weights=WEIGHTS)] == reference

    def test_identical_candidates_still_have_a_total_order(self, direct_train) -> None:
        first = make_journey(direct_train, "AAA", "CCC", journey_id="a")
        second = make_journey(direct_train, "AAA", "CCC", journey_id="b")
        assert [r.journey_id for r in sort_journeys([second, first], weights=WEIGHTS)] == ["a", "b"]
        assert [r.journey_id for r in sort_journeys([first, second], weights=WEIGHTS)] == ["a", "b"]

    def test_ranks_are_one_based_and_contiguous(self, direct_train) -> None:
        journeys = [
            make_journey(direct_train, "AAA", "CCC", journey_id=f"j{i}", fare_paise=100000 + i)
            for i in range(4)
        ]
        ordered = sort_journeys(journeys, weights=WEIGHTS)
        assert [r.rank for r in ordered] == [1, 2, 3, 4]

    def test_empty_input_returns_nothing(self) -> None:
        assert sort_journeys([], weights=WEIGHTS) == ()
        assert rank_journeys([], weights=WEIGHTS) == ()

    def test_rank_journeys_wrapper_matches_sort(self, direct_train) -> None:
        journeys = [
            make_journey(direct_train, "AAA", "CCC", journey_id=f"j{i}", fare_paise=100000 + i)
            for i in range(3)
        ]
        assert rank_journeys(journeys, weights=WEIGHTS) == tuple(
            r.journey for r in sort_journeys(journeys, weights=WEIGHTS)
        )

    def test_build_score_context_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError):
            build_score_context([])


class TestWeights:
    def test_weights_are_relative_not_absolute(self, direct_train) -> None:
        journey = make_journey(direct_train, "AAA", "CCC")
        context = build_score_context([journey])
        doubled = RankingWeights(
            availability=80,
            train_changes=30,
            fare=30,
            duration=30,
            connection_risk=20,
            separate_reservations=6,
            station_change=4,
        )
        assert (
            score_journey(journey, context=context, weights=WEIGHTS).total
            == score_journey(journey, context=context, weights=doubled).total
        )

    def test_availability_only_weights_rank_by_availability_alone(self) -> None:
        weights = RankingWeights(
            availability=100,
            train_changes=0,
            fare=0,
            duration=0,
            connection_risk=0,
            separate_reservations=0,
            station_change=0,
        )
        good = make_journey(
            make_train("YT1", [("AAA", None, "08:00", 0), ("CCC", "18:00", None, 0)]),
            "AAA",
            "CCC",
            journey_id="good",
            fare_paise=999999,
            availability=AvailabilityState.AVAILABLE,
        )
        weak = make_journey(
            make_train("YT2", [("AAA", None, "08:00", 0), ("CCC", "12:00", None, 0)]),
            "AAA",
            "CCC",
            journey_id="weak",
            fare_paise=1,
            availability=AvailabilityState.RAC,
        )
        ordered = sort_journeys([weak, good], weights=weights)
        assert [r.journey_id for r in ordered] == ["good", "weak"]

    def test_price_only_weights_rank_by_price_alone(self) -> None:
        weights = RankingWeights(
            availability=0,
            train_changes=0,
            fare=100,
            duration=0,
            connection_risk=0,
            separate_reservations=0,
            station_change=0,
        )
        cheap_slow = make_journey(
            make_train("YT1", [("AAA", None, "08:00", 0), ("CCC", "23:00", None, 0)]),
            "AAA",
            "CCC",
            journey_id="cheap",
            fare_paise=100000,
        )
        dear_fast = make_journey(
            make_train("YT2", [("AAA", None, "08:00", 0), ("CCC", "12:00", None, 0)]),
            "AAA",
            "CCC",
            journey_id="dear",
            fare_paise=900000,
        )
        ordered = sort_journeys([dear_fast, cheap_slow], weights=weights)
        assert [r.journey_id for r in ordered] == ["cheap", "dear"]

    def test_configuration_defaults_provide_weights(self) -> None:
        assert SearchConfiguration().ranking_weights == RankingWeights()

    def test_weight_json_includes_the_total(self) -> None:
        payload = WEIGHTS.to_json()
        assert payload["total_weight"] == (
            WEIGHTS.availability
            + WEIGHTS.train_changes
            + WEIGHTS.fare
            + WEIGHTS.duration
            + WEIGHTS.connection_risk
            + WEIGHTS.separate_reservations
            + WEIGHTS.station_change
        )

    def test_negative_weight_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            RankingWeights(availability=-1)


class TestRankedJourneyPayload:
    def test_to_json_exposes_score_and_journey(self, direct_train) -> None:
        journey = make_journey(direct_train, "AAA", "CCC")
        ranked = sort_journeys([journey], weights=WEIGHTS)[0]
        payload = ranked.to_json()
        assert payload["rank"] == 1
        assert payload["journey"]["journey_id"] == "journey-1"
        assert payload["journey"]["total_fare"]["amount_paise"] == 100000
        assert payload["score"]["scale"] == MAX_SUBSCORE

    def test_fare_is_serialised_as_paise(self, direct_train) -> None:
        journey = make_journey(direct_train, "AAA", "CCC", fare_paise=12345)
        ranked = sort_journeys([journey], weights=WEIGHTS)[0]
        assert ranked.to_json()["journey"]["total_fare"] == {
            "amount_paise": 12345,
            "currency": "INR",
        }
