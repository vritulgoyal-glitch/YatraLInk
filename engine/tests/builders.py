"""In-memory builders shared by the engine tests.

Builders construct journeys **directly** rather than through the fixture files,
so unit tests do not depend on the synthetic snapshot's exact numbers.  That way
the snapshot can be tuned without silently invalidating a test.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from engine.availability import Availability, AvailabilityBook, DateAgnosticAvailability
from engine.enums import AvailabilityState, JourneyType
from engine.legs import RailLeg
from engine.models import JourneyOption, JourneySegment, Station, Train, TrainStop
from engine.money import Fare
from engine.time_utils import parse_clock

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The travel date used by most fixture-based tests.
TRAVEL_DATE = date(2026, 6, 15)


def station(code: str, name: str | None = None) -> Station:
    """A station with a readable default name."""
    return Station(code=code, name=name or f"{code} Station")


def make_train(
    number: str,
    stops: list[tuple[str, str | None, str | None, int]],
    *,
    name: str | None = None,
) -> Train:
    """Build a train from ``(station, arrival, departure, day_offset)`` tuples."""
    return Train(
        number=number,
        name=name or f"Test Train {number}",
        stops=tuple(
            TrainStop(
                station_code=station_code,
                sequence=index,
                arrival=parse_clock(arrival) if arrival else None,
                departure=parse_clock(departure) if departure else None,
                day_offset=day_offset,
            )
            for index, (station_code, arrival, departure, day_offset) in enumerate(stops, start=1)
        ),
    )


def make_leg(
    train: Train, origin: str, destination: str, anchor_date: date = TRAVEL_DATE
) -> RailLeg:
    """Build a leg of ``train`` from ``origin`` to ``destination``."""
    return RailLeg(
        train=train,
        origin_index=train.stop_index(origin),
        destination_index=train.stop_index(destination),
        anchor_date=anchor_date,
    )


def make_segment(
    train: Train,
    origin: str,
    destination: str,
    *,
    anchor_date: date = TRAVEL_DATE,
    availability: AvailabilityState = AvailabilityState.AVAILABLE,
    fare_paise: int | None = 100000,
    reservation_index: int = 0,
) -> JourneySegment:
    """Build a validated journey segment from a leg."""
    leg = make_leg(train, origin, destination, anchor_date)
    return JourneySegment(
        train_number=leg.train_number,
        train_name=leg.train.name,
        origin_station_code=leg.origin_station_code,
        destination_station_code=leg.destination_station_code,
        departure=leg.departure,
        arrival=leg.arrival,
        duration_minutes=leg.duration_minutes,
        availability=availability,
        fare=Fare.from_paise(fare_paise) if fare_paise is not None else None,
        reservation_index=reservation_index,
        boarding_stop_sequence=leg.boarding_sequence,
        alighting_stop_sequence=leg.alighting_sequence,
    )


def make_journey(
    train: Train,
    origin: str,
    destination: str,
    *,
    journey_id: str = "journey-1",
    anchor_date: date = TRAVEL_DATE,
    availability: AvailabilityState = AvailabilityState.AVAILABLE,
    fare_paise: int | None = 100000,
    journey_type: JourneyType = JourneyType.DIRECT,
) -> JourneyOption:
    """Build a single-segment journey, convenient for ranking tests."""
    segment = make_segment(
        train,
        origin,
        destination,
        anchor_date=anchor_date,
        availability=availability,
        fare_paise=fare_paise,
    )
    return JourneyOption(
        journey_id=journey_id,
        journey_type=journey_type,
        origin_station_code=origin,
        destination_station_code=destination,
        segments=(segment,),
    )


def make_book(
    entries: list[Availability | DateAgnosticAvailability],
) -> AvailabilityBook:
    """Build an availability book from explicit entries."""
    return AvailabilityBook(entries)


def at(day: date, hour: int, minute: int = 0) -> datetime:
    """Convenience datetime builder for tests."""
    return datetime(day.year, day.month, day.day, hour, minute)


def journeys_of_type(result, journey_type: JourneyType) -> tuple[JourneyOption, ...]:
    """All returned journeys of one type, best ranked first."""
    return tuple(journey for journey in result.options if journey.journey_type is journey_type)


def find_journey(
    result, *, train_numbers: tuple[str, ...], journey_type: JourneyType | None = None
) -> JourneyOption | None:
    """Find a returned journey by its exact ordered train sequence."""
    for journey in result.options:
        if journey.train_numbers != train_numbers:
            continue
        if journey_type is not None and journey.journey_type is not journey_type:
            continue
        return journey
    return None
