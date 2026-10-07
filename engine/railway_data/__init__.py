"""The normalized railway data contract (Phase 2A).

Any future *authorised* railway data provider maps its own response format into
the models in this package; the Phase 1 journey engine consumes only these
models, so it never depends on a provider's payload shape.

This package contains no network access, no API keys, no database code, no
provider-specific parsing, no scraping and no booking logic.  See
``engine/railway_data/README.md``.
"""

from __future__ import annotations

from engine.railway_data.models import (
    AvailabilitySnapshot,
    Quota,
    RailwayStationStop,
    Station,
    Train,
    TrainClass,
    TrainSchedule,
    coerce_quota,
    normalize_quota,
    normalize_train_class,
)
from engine.railway_data.validation import (
    as_tzinfo,
    localise,
    require_aware_datetime,
    require_non_empty_text,
    require_optional_count,
    require_ordered_route,
    require_paise,
    require_schedule_date,
    require_timezone_name,
    require_unique_stations,
    schedule_minutes,
    validate_day_offsets,
    validate_sequence,
    validate_state_counters,
    validate_stop_times,
)

__all__ = [
    "AvailabilitySnapshot",
    "Quota",
    "RailwayStationStop",
    "Station",
    "Train",
    "TrainClass",
    "TrainSchedule",
    "as_tzinfo",
    "coerce_quota",
    "localise",
    "normalize_quota",
    "normalize_train_class",
    "require_aware_datetime",
    "require_non_empty_text",
    "require_optional_count",
    "require_ordered_route",
    "require_paise",
    "require_schedule_date",
    "require_timezone_name",
    "require_unique_stations",
    "schedule_minutes",
    "validate_day_offsets",
    "validate_sequence",
    "validate_state_counters",
    "validate_stop_times",
]
