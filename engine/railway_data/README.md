# Normalized Railway Data Contract (Phase 2A)

This package is the **boundary between railway data and the YatraLink engine**.

```
External railway provider
          |
  provider adapter        <-- Phase 2B (not built yet)
          |
  THIS CONTRACT           <-- Phase 2A: engine/railway_data/
          |
Phase 1 journey engine
          |
Ranked journey alternatives
```

The engine is a deterministic computation over railway facts. It must never know
how a provider formats a response, what its field names are, which quota code it
uses for Tatkal, or whether the inventory came back as `"AVAIL"` or `"available"`.
All of that is the adapter's job. What crosses the boundary is the small,
provider-neutral vocabulary defined here.

> **Phase 2A scope.** Models, validation and documentation only. There are **no
> live HTTP calls, no API keys, no database or Redis integration, no scraping, no
> IRCTC automation, no CAPTCHA bypass, no booking, no payment and no
> provider-specific parsing** in this package. Phase 2B will map one *authorised*
> provider into these models.

---

## 1. Why this contract exists

Without it, every provider format would leak into the engine: journey generation
would start branching on a vendor's JSON shape, and the deterministic behaviour
Phase 1 tested would depend on which source answered. With it:

* the engine consumes **one** set of types, no matter who supplied the data;
* a provider adapter has an explicit, testable target to map into;
* provider provenance and freshness are transported, not guessed;
* the `UNKNOWN` rule from Phase 1 survives the boundary intact — a provider that
  reports nothing can never produce an `AVAILABLE` journey.

The contract is deliberately small: seven concepts, plus focused validation.
There are no speculative abstractions, no plugin registry, no async client, and
no dependency beyond the Python standard library.

---

## 2. The normalized models

| Model | Fields | Notes |
| --- | --- | --- |
| `Station` | `code`, `name`, optional `city`, optional `timezone` | `code` is normalised with Phase 1's rules and is the identity used everywhere. `name` is display text and is **not** assumed unique. `timezone` is an IANA name such as `Asia/Kolkata`. |
| `RailwayStationStop` | `station_code`, `arrival`, `departure`, `day_offset`, `sequence`, (`departure_day_offset`) | One scheduled stop. Clock times are naive, minute-granular wall-clock times; `day_offset` is preserved exactly. |
| `Train` | `train_number`, `train_name`, `stops` | The ordered route, exactly as reported. The route is the only source of truth for direction. |
| `TrainClass` | normalised travel-class identifier | **Phase 1's `engine.enums.TravelClass`**, re-exported. `SLEEPER = "SL"`, `AC_3_TIER = "3A"`, … No duplicate enum. |
| `Quota` | normalised quota identifier | `GENERAL`, `TATKAL`, `PREMIUM_TATKAL`, `LADIES`, `SENIOR_CITIZEN`, `DIVYAANG`. Phase 1 stores a quota as a free string, so this enum only *names* the values the engine already accepts (`Quota.GENERAL.value` is Phase 1's `DEFAULT_QUOTA`). |
| `TrainSchedule` | `train`, `service_date`, ordered `stops` | A train's routine on one calendar date. `service_date` is the day on which `day_offset == 0` begins. |
| `AvailabilitySnapshot` | see below | The reported inventory for **one bookable unit**: one train / travel date / class / quota / origin–destination. |

### `AvailabilitySnapshot`

Required: `train_number`, `origin_station_code`, `destination_station_code`,
`travel_date`, `travel_class`, `provider`, `fetched_at`.

Optional: `state` (defaults to `UNKNOWN`), `quota` (defaults to `GENERAL`),
`available_count`, `rac_count`, `waitlist_value`, `fare_paise`, `updated_at`.

Two rules make it safe:

* **Counters must match the state.** `available_count` is only meaningful for
  `AVAILABLE`, `rac_count` only for `RAC`, `waitlist_value` only for `WAITLIST`,
  and `NOT_AVAILABLE` / `UNKNOWN` may carry none. So an adapter cannot return
  `UNKNOWN` *plus* `available_count = 1` and smuggle an availability claim
  through the contract — that is an error, not a hint.
* **A reported counter is at least 1.** "`AVAILABLE` with 0 seats" is a
  contradiction; report `NOT_AVAILABLE` instead.

`AvailabilitySnapshot.unreported(...)` builds the "the source had nothing to say"
snapshot, which is always `UNKNOWN` with no counters and no fare.

### Money

Fares are an integer number of **paise** (`fare_paise: int`). A `float`, a `bool`
or any non-integer is **rejected**, never rounded, and a negative amount is
rejected too. `AvailabilitySnapshot.fare` returns Phase 1's exact
`engine.money.Fare`, so nothing downstream ever handles a floating-point price.

### Time

Two different kinds of time are kept strictly apart:

* **Schedule times** are naive, minute-granular wall-clock times plus a
  `day_offset`. `RailwayStationStop.arrival_at(service_date, timezone)` and
  `TrainSchedule.arrival_at(station, timezone)` *localise* the
  `service_date + day_offset` day with the station's timezone — they never
  convert. So a `21:40` departure on day 0 is still `21:40`, an arrival at
  `04:30` with `day_offset = 1` is still `04:30` on the **next** calendar day,
  and nothing is silently absorbed into a timezone shift.
* **Instants** — `fetched_at` and `updated_at` — must be **timezone-aware**. A
  naive datetime does not identify an instant, so it is rejected rather than
  assumed to be UTC. `updated_at` must not be later than `fetched_at`.

`day_offset` is non-negative and never decreases along a route, so the ordered
stop sequence and its day offsets can never contradict each other.

---

## 3. Validation

All invariants live in `engine/railway_data/validation.py`, so the models stay
declarative and an adapter can read the rules in one file:

| Rule | Enforced by |
| --- | --- |
| Non-empty train number (Phase 1's normalisation) | `engine.models.normalize_train_number` |
| Valid station code (uppercase, `[A-Z][A-Z0-9]{1,9}`) | `engine.models.normalize_station_code` |
| Positive sequence (≥ 1) | `RailwayStationStop` |
| Strictly increasing sequence, no duplicates | `validate_sequence` |
| No station visited twice on a route | `require_unique_stations` |
| Origin occurs **before** destination | `require_ordered_route` (route reversal is rejected, never applied silently) |
| Arrival/departure consistency | `validate_stop_times` |
| Valid `day_offset` (≥ 0, never decreasing) | `validate_day_offsets`, `validate_stop_times` |
| Non-negative integer paise fare | `require_paise` |
| Valid travel class | `engine.enums.coerce_travel_class` |
| Valid quota | `coerce_quota` |
| Availability state / counter consistency | `validate_state_counters` |

---

## 4. `UNKNOWN` semantics (unchanged from Phase 1)

| Situation | Contract result |
| --- | --- |
| The source reported `AVAILABLE` | `AVAILABLE` (the only confirmed state) |
| The source reported nothing | `UNKNOWN` via `AvailabilitySnapshot.unreported(...)` or `state=UNKNOWN` |
| The adapter could not find a unit | `None` from the provider, which the engine renders as `UNKNOWN` |

**Missing external availability never becomes `AVAILABLE`.** `is_confirmed` is
`True` only for a reported `AVAILABLE`, and an `UNKNOWN` snapshot may not carry
an availability counter, so there is no route by which silence turns into
inventory. Phase 1's aggregation ladder
(`AVAILABLE < RAC < WAITLIST < UNKNOWN < NOT_AVAILABLE`) applies unchanged.

---

## 5. Provider adapter responsibility (Phase 2B)

An adapter owns **transport and mapping, nothing else**:

1. retrieve the payload from the *authorised* source, using its documented
   authentication;
2. map its fields into the models above — station codes, train numbers, class
   codes, quota codes, clock times, day offsets, paise amounts;
3. supply provenance: `provider`, `fetched_at`, and `updated_at` when (and only
   when) the source reports it;
4. signal "nothing reported" as `None` (or an `UNKNOWN` snapshot) — never as
   `AVAILABLE`;
5. hand the snapshots to the engine exactly where Phase 1 already expects them.

The last point needs no engine change: Phase 1 already defines the seam.
`engine.provider.AvailabilityProvider` is the protocol an adapter implements, and
`AvailabilitySnapshot.to_availability_record()` returns the
`engine.provider.AvailabilityRecord` Phase 1's boundary transports. So Phase 2B
is a new adapter module plus a class that implements `get_availability(query)`,
plugged in through the existing `ProviderAvailabilityBook` — with **no change to
generation, validation, ranking or aggregation**.

Deliberately *not* the adapter's job: deciding whether two stations are the same
place, judging whether a transfer is feasible, aggregating multi-segment
availability, ranking journeys, inventing a missing fare or a missing state, or
re-interpreting `UNKNOWN`.

---

## 6. Usage

```python
from datetime import date, datetime, timezone

from engine.railway_data import AvailabilitySnapshot, Quota, TrainClass

snapshot = AvailabilitySnapshot(
    train_number="YT1001",
    origin_station_code="BLR",
    destination_station_code="DEL",
    travel_date=date(2026, 7, 1),
    travel_class=TrainClass.AC_3_TIER,
    quota=Quota.GENERAL,
    state="AVAILABLE",
    available_count=42,
    fare_paise=245000,
    provider="authorised-source-x",
    fetched_at=datetime(2026, 6, 30, 6, 15, tzinfo=timezone.utc),
)

record = snapshot.to_availability_record()  # engine.provider.AvailabilityRecord
```

Anything invalid raises `engine.errors.DomainValidationError`, which is also a
`ValueError` — the same error type the rest of the engine already uses.

---

## 7. What this contract is not

* It is not a provider client: there is no HTTP, no retry, no auth, no key.
* It is not a cache, a database schema or a search index.
* It is not a booking surface: no reservations, no payments, no accounts.
* It contains no provider-specific parsing and no scraping.
* It does not redefine anything Phase 1 already models — travel class,
  availability state, fares, station/train codes and stop arithmetic all come
  from Phase 1.
