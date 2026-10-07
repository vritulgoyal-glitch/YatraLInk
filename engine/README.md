# YatraLink Connection Engine

This directory contains the deterministic railway journey-optimization engine.

The engine answers one question: **given a railway snapshot and a traveller's
request, which journeys are actually possible, and which are best?** It never
invents railway facts, never calls an LLM, and never touches the network.

> Phase 1 scope: generate, validate, deduplicate and rank journeys from
> **synthetic fixture data**. No IRCTC integration, no scraping, no CAPTCHA
> bypass, no personal-account automation. The fixture layer is designed so a
> future authorised railway-data adapter can replace it without changing a single
> algorithm — see [Fixture usage](#fixture-usage).

---

## 1. Responsibilities (Phase 1)

- **Domain models** — stations, trains, ordered routes, stops with day offsets,
  journey segments, journey options, connections.
- **Timetable representation** — an immutable, index-backed network snapshot.
- **Search requests** — validated and normalised once, at construction.
- **Availability representation** — reported inventory plus a documented
  aggregation rule; `UNKNOWN` is never treated as confirmed.
- **Journey generation** — direct, same-train split, and connecting candidates
  (maximum 2 train changes / 3 segments).
- **Connection validation** — real datetime arithmetic with a configurable
  minimum transfer buffer.
- **Connection risk** — deterministic `SAFE` / `TIGHT` / `INVALID`
  classification.
- **Journey duration** — `final arrival − first departure` (elapsed time, not the
  sum of segment durations).
- **Fare aggregation** — exact integer paise, never floating point.
- **Candidate deduplication** — on journey type + ordered train/station structure
  + departure/arrival structure.
- **Deterministic ranking** — an explainable, weight-configurable integer score.
- **Tests and fixtures** — a full pytest suite plus synthetic sample data.

The engine is deliberately independent of FastAPI, Next.js, Supabase, Redis,
external railway providers and LLMs. A future service layer simply calls
`engine.generate_journey_options(...)`.

---

## 2. Module map

| Module | Responsibility |
| --- | --- |
| `engine.models` | Domain entities: `Station`, `Train`, `TrainStop`, `JourneySegment`, `ConnectionInfo`, `JourneyOption`; station/train-code normalisation |
| `engine.enums` | `AvailabilityState`, `JourneyType`, `ConnectionKind`, `ConnectionRisk`, `TravelClass`, `RejectionReason`, and the documented desirability ladders |
| `engine.errors` | Explicit domain errors (`DomainValidationError`, `InvalidJourneyError`, `UnknownStationError`, `DataSourceError`, `CandidateLimitError`) |
| `engine.money` | `Fare` in integer paise; `float` amounts are rejected |
| `engine.time_utils` | Day-offset aware schedule arithmetic, clock parsing/formatting, duration helpers |
| `engine.network` | `Timetable` snapshot and directional `TransferAllowance` declarations |
| `engine.search` | `SearchRequest` with full validation |
| `engine.availability` | `Availability`, date-independent snapshot entries, `AvailabilityBook`, aggregation |
| `engine.legs` | `RailLeg` — one ride on one train between two of its stops |
| `engine.candidates` | Structural generation: direct, same-train split, connecting |
| `engine.connections` | Transfer assessment and risk classification |
| `engine.journey` | Plan → validated, priced option; journey-level validation |
| `engine.ranking` | Explainable scoring, tiebreakers, deterministic ordering |
| `engine.pipeline` | End-to-end generate → validate → deduplicate → rank |
| `engine.fixtures` | Loading synthetic JSON snapshots into engine structures |

---

## 3. Domain models

### `Station`
`code` (normalised, uppercase, the identity used everywhere) and `name`
(`code`-unique, `name`-**not**-unique: two different stations may share a name).

### `Train`
`number`, `name`, and an ordered tuple of `TrainStop`. The route is the only
source of truth for direction: a train serves `origin → destination` **only** when
origin appears strictly before destination in the route. **Routes are never
reversed by the engine.** Duplicate station visits, non-increasing stop
sequences and non-chronological stop times are rejected at construction.

### `TrainStop`
`station_code`, 1-based `sequence`, `arrival` / `departure` clock times,
`day_offset`, and optional `departure_day_offset` (equal to `day_offset` or
`day_offset + 1`, which is how a halt that itself crosses midnight is expressed).
The origin stop needs only a departure; the terminus needs only an arrival;
intermediate stops need both.

### `Fare`
`amount_paise: int` (plus a currency code). `Fare.from_rupees` accepts `Decimal`,
`int` or a decimal string and rejects `float`. Arithmetic is exact; any amount that
cannot be represented in whole paise is rejected rather than rounded.

### `JourneySegment`
One reservation-sized leg on one train: train, origin, destination, departure,
arrival, duration, availability, optional fare and seats, plus
`reservation_index` so a same-train split's separate tickets are explicit. A
segment whose declared duration disagrees with its own timestamps is rejected.

### `JourneyOption`
`journey_id`, `journey_type`, origin, destination, ordered `segments`, ordered
`connections`, and an `availability_summary`. Derived, read-only values:
`departure`, `arrival`, `total_duration_minutes`, `total_travel_minutes`,
`total_transfer_minutes`, `total_fare`, `train_changes`, `reservation_count`,
`requires_separate_reservations`, `availability`, `risk`, `is_confirmed`,
`station_path`, `dedup_key` and `stable_sort_key`.

---

## 4. Journey types

| Type | Meaning | Train changes |
| --- | --- | --- |
| `DIRECT` | One train, one reservation, origin → destination in route order | `0` |
| `SAME_TRAIN_SPLIT` | Two or three reservations on the **same physical train** (e.g. BLR→HYD then HYD→DEL) | `0` |
| `CONNECTING` | At least one physical change of train | `≥ 1` |

A same-train split is **not** automatically safe or equivalent to a direct
booking: `requires_separate_reservations` is `True`, each reservation's inventory
and fare are evaluated independently, and the journey may therefore aggregate to
`WAITLIST` even though the passenger never leaves the train.

**Depth limit:** at most 2 train changes, therefore at most 3 segments. This is
enforced by `SearchConfiguration` and by a hard guard in `JourneyOption`, and the
connecting search is a bounded expansion rather than a general path search.

---

## 5. Availability states

| State | Meaning |
| --- | --- |
| `AVAILABLE` | Bookable inventory reported by the data source |
| `RAC` | Reservation Against Cancellation |
| `WAITLIST` | Waitlisted inventory |
| `UNKNOWN` | The data source reported nothing — **never** confirmed |
| `NOT_AVAILABLE` | No inventory for the requested segment/class/quota |

### Aggregation

The journey's state is the **worst** state on this ladder:

```
AVAILABLE (0)  <  RAC (1)  <  WAITLIST (2)  <  UNKNOWN (3)  <  NOT_AVAILABLE (4)
```

Two deliberate, documented consequences:

- `UNKNOWN` ranks *below* `WAITLIST` (unknown inventory is not actionable) but
  *above* `NOT_AVAILABLE` (nothing was reported as impossible).
- `is_confirmed` is `True` **only** when every required segment is `AVAILABLE`.
  A `UNKNOWN` segment never yields a confirmed journey, and if any segment fare is
  unknown the journey's total fare is `None` — not `0`.

By default a journey containing a `NOT_AVAILABLE` segment is rejected before
ranking, because it cannot be travelled. Set
`include_not_available_journeys=True` to return such journeys explicitly flagged
instead.

---

## 6. Connection rules

Configuration lives in `engine.config.SearchConfiguration`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `minimum_connection_minutes` | `30` | Required transfer time between two different trains at one station |
| `tight_connection_max_buffer_minutes` | `30` | A valid connection with at most this much spare buffer is `TIGHT` |
| `cross_station_minimum_minutes` | `90` | Default requirement for a station change |
| `allow_cross_station_transfers` | `False` | Station changes are off unless explicitly enabled *and* declared by the data |
| `max_train_changes` | `2` | Hard cap |
| `max_segments` | `3` | Hard cap |
| `same_train_split_max_segments` | `2` | How far a split may be subdivided |
| `max_candidates_per_type` | `4000` | Safety valve against combinatorial explosion |
| `include_not_available_journeys` | `False` | Whether `NOT_AVAILABLE` journeys are returned |

### Classification

`SAME_TRAIN`
: The passenger does not leave the train. Requirement is **0 minutes**; the result
  is `SAFE`. Separate tickets are still required.

`CROSS_TRAIN_SAME_STATION`
: Requirement is `minimum_connection_minutes`. With
  `buffer = transfer − required`:
  * `buffer < 0` → `INVALID`
  * `0 ≤ buffer ≤ tight_connection_max_buffer_minutes` → `TIGHT`
  * `buffer > tight_connection_max_buffer_minutes` → `SAFE`

  So a 20-minute transfer against a 30-minute minimum is `INVALID`, a 30-minute
  transfer is valid but `TIGHT` (buffer 0), and a 60-minute transfer is `SAFE`
  (buffer 30).

`CROSS_STATION_TRANSFER`
: Only considered when `allow_cross_station_transfers` is enabled **and** the data
  declares a `TransferAllowance` for that **ordered** station pair. Station
  changes are directional — `NDLS → DEL` does not imply `DEL → NDLS`. The
  requirement is `max(allowance.minimum_minutes, minimum_connection_minutes)`, and
  the risk is **capped at `TIGHT`**: a station change is never `SAFE`.

The engine never assumes two distinct station codes are the same place. `NDLS`
and `DEL` are different stations unless the data says otherwise.

All connection arithmetic uses real datetimes, never raw clock times, so
midnight-crossing and multi-day connections are exact.

---

## 7. Ranking philosophy

Ranking is a pure function of the candidate set and the configured weights: no AI,
no randomness, no clock, no I/O, no floating point.

Each journey receives seven integer sub-scores on a `0..1000` scale (**higher is
better**) and a weighted mean as its total:

```
total = sum(weight_i * subscore_i) // sum(weight_i)      # 0..1000, integer
```

| Component | Scores 1000 when | Default weight |
| --- | --- | --- |
| `availability` | every segment is `AVAILABLE` | 40 |
| `train_changes` | direct (no change of train) | 15 |
| `fare` | lowest total fare in the candidate set | 15 |
| `duration` | shortest elapsed time in the candidate set | 15 |
| `connection_risk` | all connections are `SAFE` | 10 |
| `separate_reservations` | a single reservation is needed | 3 |
| `station_change` | no station change at any transfer | 2 |

Fare and duration have no absolute maximum, so they are normalised against the
**observed range in this candidate set** (`normalise()`), using integer
arithmetic. When every candidate shares a value, that component scores 1000 for
everyone — it neither rewards nor penalises. An **unknown fare scores 0**, so an
unpriced journey can never out-rank a genuinely cheaper priced one.

Weights are *relative*: only their ratios matter, because the total is normalised
by their sum. "Optimise for time" or "optimise for price" is therefore a
configuration change, not a code change. Every score is explainable through
`JourneyScore.explain()` / `to_json()`, which report each component's weight,
sub-score, weighted contribution and a human-readable detail string.

### Tiebreakers (documented and stable)

When two totals are equal, `compare_journeys` applies, in order:
`DIRECT` → `SAME_TRAIN_SPLIT` → `CONNECTING`; then `SAFE` → `TIGHT` risk; better
availability; fewer train changes; lower fare (unknown fare last); shorter
duration; later departure; and finally `stable_sort_key` (journey type, train
numbers, station path, departure, arrival, journey id).

Because ranking depends only on the *set* of candidates, shuffling the input
cannot change the order, and identical candidates always appear in the same
relative order.

---

## 8. Pipeline

`engine.pipeline.generate_journeys(...)` runs the whole chain:

1. the request was already validated by `SearchRequest`;
2. station codes are normalised by the model layer;
3. the request's stations are checked against the timetable (a missing station
   raises rather than silently returning nothing);
4. structural candidates are generated — direct, then same-train splits, then
   connecting;
5. each candidate is assembled and validated into a priced `JourneyOption`, or
   recorded as a `JourneyRejection` with a machine-readable `RejectionReason` and
   a human-readable detail;
6. duplicates are removed and recorded;
7. the survivors are ranked;
8. the ranked list is returned, optionally truncated with `limit`.

Nothing is dropped silently — the result carries
`generated_plan_count`, `candidate_count`, `duplicate_count`, `type_counts`,
`availability_counts`, `rejection_counts` and every rejection.

### Interface

```python
from datetime import date
from engine import SearchRequest, generate_journey_options

request = SearchRequest(origin="BLR", destination="DEL", travel_date=date(2026, 6, 15))

options = generate_journey_options(
    request,
    timetable=network.timetable,
    availability_book=network.availability_book,
    configuration=network.configuration,
    transfer_allowances=network.transfer_allowances,
)

for option in options:
    print(option.journey_type, option.total_duration_minutes, option.total_fare)
```

Use `generate_journeys(...)` (same arguments) when the audit trail —
rejections, counts, scores — is also needed; it returns a
`JourneyGenerationResult`. `generate_journeys_for_network(request, network)` is a
convenience wrapper for a loaded `FixtureNetwork`.

### Where the request's own limits are applied

`earliest_departure` and `latest_arrival` are applied **during generation** as a
sound prune: the search starts at the earliest acceptable departure, and a leg
that would arrive after the deadline is never created, because adding further
segments can only make a journey longer. This keeps the candidate set bounded.
Journey-level validation re-checks both constraints defensively, so a candidate
that somehow missed the prune is still rejected with
`RejectionReason.AFTER_LATEST_ARRIVAL` / `BEFORE_EARLIEST_DEPARTURE` rather than
being returned. `generated_plan_count` reports how many plans survived the prune,
so the effect of a deadline is visible in the result.

---

## 9. Fixture usage

Fixtures live in `data/fixtures/` as plain JSON:

| File | Contents |
| --- | --- |
| `network.json` | Snapshot name, description, scenario list, `search_configuration` |
| `stations.json` | Station codes and display names |
| `trains.json` | Train numbers, names and ordered stops (with day offsets) |
| `availability.json` | Reported inventory and fares in paise |
| `transfer_allowances.json` | Explicit directional different-station allowances |

```python
from engine.fixtures import load_fixture_network, default_network

network = load_fixture_network()  # shipped snapshot (cached)
other = load_fixture_network("/path/to/dir")  # any directory with the same files
```

The loader is the **only** part of the engine that knows about files. It produces
a `FixtureNetwork` — a timetable, an availability book, transfer allowances and a
configuration — which is exactly what the pipeline consumes. A future
railway-data adapter can therefore replace this module without any algorithm
changing.

An availability entry with `"travel_date": null` is a
`DateAgnosticAvailability`: the snapshot declares that state regardless of travel
date. A dated entry always takes precedence. A segment with **no** record is
reported as `UNKNOWN` with an unknown fare — the engine never invents one.

> **The shipped fixtures are synthetic sample data.** They are not Indian Railways
> schedules, the train numbers are invented and prefixed `YT`, and the trains are
> named "Sample …" so this cannot be mistaken for real timetable data.

---

## 10. Running the tests

The engine tests live in `engine/tests/` and are discoverable from the backend,
which is where the project's test command runs:

```bash
cd services/api
pytest
```

That single command runs the backend's `tests/` **and** `engine/tests/`
(`services/api/pyproject.toml` sets `testpaths`, and `pythonpath` puts the
repository root on `sys.path` so `import engine` resolves). To run the engine
suite alone from the repository root:

```bash
pytest engine/tests
```

The suite is organised by concern: `test_time`, `test_models`,
`test_search_request`, `test_fixtures`, `test_direct`, `test_same_train_split`,
`test_connections`, `test_invalid_journeys`, `test_availability`, `test_ranking`
and `test_pipeline`. The last also asserts the cross-component boundary: the
engine imports no `app` package, no web framework, no HTTP/database/Redis client,
no LLM library, performs no network calls and contains no secrets.

Linting, when configured, uses Ruff (`services/api/pyproject.toml`, line length
100, targeting py311).

---

## 11. Design notes and non-goals

**No AI in the engine.** Not for feasibility, station ordering, arrival/departure
validity, connection validity, transfer duration, journey duration, fare,
availability state, train changes, or ranking facts. AI will be an explanation
layer in a later phase, consuming this engine's structured output.

**No data invention.** The engine operates only on supplied data. Missing
schedules, availability or fares are represented explicitly as absent, never
guessed.

**Deterministic by construction.** Every collection the engine iterates is
ordered deterministically (stations by code, trains by number, legs by train then
route position, combinations in route order), so identical inputs always produce
identical output.

**Deliberate Phase 1 simplifications** (all documented at the point of use):

- Trains are modelled as running **daily**; for each train at a station only the
  earliest occurrence at or after the relevant threshold is considered. Running
  days / days-of-operation are future work.
- Times are **naive local** railway times; timezones and daylight saving are
  intentionally out of scope.
- Maximum 2 train changes / 3 segments; no arbitrary-depth path search.
- No FastAPI endpoints for search — the engine is independently testable, and
  Phase 1 keeps the existing `/health` endpoint working.

---

## 12. The availability-provider boundary

The engine consumes availability through a **data contract**, never through the
fixture implementation. `engine/provider.py` defines it:

| Name | Kind | Responsibility |
| --- | --- | --- |
| `AvailabilityProvider` | `typing.Protocol` (runtime-checkable) | What every data source implements: `get_availability(query) -> AvailabilityRecord \| None` plus a `provider_name`. **Data access only** — no journey logic. |
| `AvailabilityQuery` | frozen dataclass | One bookable unit asked about: train / origin / destination / travel date / class / quota. |
| `AvailabilityRecord` | frozen dataclass | What a provider reports: the five states, seats, fare, plus provenance (`provider`) and freshness (`fetched_at`, `updated_at`). |
| `AvailabilityLookup` | `typing.Protocol` | What the engine consumes. `AvailabilityBook` already satisfies it structurally. |
| `ProviderAvailabilityBook` | adapter | Wraps any `AvailabilityProvider` and presents it as an engine-facing lookup, so `generate_journeys` / `generate_journey_options` accept a provider-backed book unchanged. |

`engine/fixture_provider.py` contains `FixtureAvailabilityProvider`, the Phase 1
implementation of the protocol. It reads the synthetic snapshot (via
`AvailabilityBook`) and returns `AvailabilityRecord` values — nothing more.
A future live railway provider implements the same protocol; no engine
algorithm changes, and the engine never learns whether data came from JSON, a
database or an API.

**Freshness metadata.** `fetched_at` / `updated_at` are transported verbatim on
the record (and `source_updated_at` on the domain `Availability`, preferring
`updated_at` and falling back to `fetched_at`). No staleness policy is applied:
stale data is **not** automatically downgraded to `UNKNOWN` — that is a future
policy decision, deliberately not made yet.

**UNKNOWN semantics.** A provider signals "the source did not report this unit"
by returning `None`, which the engine renders as `AvailabilityState.UNKNOWN`.
An explicit `UNKNOWN` record behaves identically. `UNKNOWN` is never confirmed:
only `AVAILABLE` is (`AvailabilityState.is_confirmed`), and a journey is
confirmed only when every segment is `AVAILABLE`.
