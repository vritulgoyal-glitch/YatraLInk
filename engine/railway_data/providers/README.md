# Development Railway-Data Providers (Phase 2B)

Adapter-style providers that map a railway data source into the **Phase 2A
normalized contract** (`engine/railway_data`), so the Phase 1 journey engine can
consume the data without ever seeing a provider-specific format:

```
Railway timetable data
        ↓
Provider adapter            <- this package (Phase 2B)
        ↓
Phase 2A normalized models  <- engine/railway_data (Station, Train,
        ↓                      RailwayStationStop, TrainSchedule)
Phase 1 journey engine      <- unchanged algorithms
```

## ⚠️ Development-only, by design

The provider shipped here (`timetable_fixture.py`) is a **development timetable
fixture provider**:

* **This is development-only timetable data.** The fixture lives in
  `data/fixtures/timetable/timetable.json` and is marked
  `DEVELOPMENT_FIXTURE_DATA` in its own metadata block. The provider refuses to
  load a file that is not marked that way.
* **It is not live railway data.** Train numbers are prefixed `YTF`
  (YatraLink Fixture). Times, stop sequences and day offsets are synthetic and
  do **not** represent current — or any real — railway schedules.
* **It is not an IRCTC API.** There is no IRCTC connection, no IRCTC data, no
  scraping, no automation, no CAPTCHA interaction, no credentials and no
  network access of any kind in this package.
* **It must not be presented to users as current availability.** The provider
  implements timetable lookups only — it never produces an availability, fare,
  RAC/WL or quota claim of any kind. The Phase 2A `AvailabilitySnapshot` is
  never constructed here, and every provider record carries the provenance name
  `development-timetable-fixture`.
* **The production provider will replace this adapter.** Once YatraLink signs
  an agreement granting it data-display rights (the Phase 2B due-diligence
  outcome: an authorised provider, conditional on a signed contract), the
  fixture adapter is swapped for the authorised one behind the same Phase 2A
  models. Nothing above the adapter changes.
* **The Phase 1 engine does not depend on provider-specific formats.** The
  engine consumes only the Phase 2A contract models; this adapter is the only
  place that knows the fixture's JSON shape.

## Interface

`TimetableFixtureProvider` exposes timetable responsibility #1 only
(seat availability stays with the existing
`engine.fixture_provider.FixtureAvailabilityProvider` path — the two concerns
are deliberately kept separate, and the `AvailabilityProvider` protocol is not
touched):

| Method | Returns | Not-found behaviour |
| --- | --- | --- |
| `get_station(code)` | `Station \| None` | `None` for unknown stations |
| `get_train(train_number)` | `Train \| None` | `None` for unknown trains |
| `get_schedule(train_number, service_date)` | `TrainSchedule \| None` | `None`; never invents a schedule |
| `trains_between_stations(origin, destination, service_date)` | `TrainsBetweenStationsResult` | empty `trains` for unknown/reversed/equal pairs; routes are never reversed |

All results are deterministic. `trains_between_stations` orders matches by
origin departure minute, then train number, and preserves the queried
`service_date` verbatim on the result.

## Data rules honoured by the fixture

* Station codes are normalized with Phase 1's rules (`"blr"` finds `BLR`).
* Stop `sequence` values are 1-based and strictly increasing.
* `day_offset` is **written explicitly in the data** and never inferred from
  the naive clock times — an arrival after midnight carries `day_offset: 1`,
  and a halt that itself crosses midnight uses `departure_day_offset`.
* Every record is constructed through the Phase 2A models, so all Phase 2A
  validation (sequence monotonicity, chronology, unique stations, clock rules,
  IANA timezones) runs at load time. The adapter duplicates no validation.
