"""Shared fixtures for the engine test suite."""

from __future__ import annotations

import pytest

from engine.availability import AvailabilityBook
from engine.fixtures import FixtureNetwork, load_fixture_network
from engine.tests.builders import (
    TRAVEL_DATE,
    at,
    find_journey,
    journeys_of_type,
    make_book,
    make_journey,
    make_leg,
    make_segment,
    make_train,
    station,
)

__all__ = [
    "TRAVEL_DATE",
    "at",
    "availability_book",
    "config",
    "find_journey",
    "journeys_of_type",
    "make_book",
    "make_journey",
    "make_leg",
    "make_segment",
    "make_train",
    "network",
    "station",
    "timetable",
]


@pytest.fixture(scope="session")
def network() -> FixtureNetwork:
    """The shipped synthetic fixture network, loaded once per session."""
    return load_fixture_network()


@pytest.fixture()
def timetable(network: FixtureNetwork):
    """The fixture timetable."""
    return network.timetable


@pytest.fixture()
def availability_book(network: FixtureNetwork) -> AvailabilityBook:
    """The fixture availability snapshot."""
    return network.availability_book


@pytest.fixture()
def config(network: FixtureNetwork):
    """The fixture's engine configuration."""
    return network.configuration
