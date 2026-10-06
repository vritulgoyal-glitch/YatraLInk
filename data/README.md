# Railway Data Layer

The data layer will contain normalized railway data models and ingestion
contracts.

Planned entities:

- Stations
- Trains
- Train routes
- Train classes
- Availability
- Fares
- Data-source metadata

Availability must distinguish AVAILABLE, RAC, WAITLIST, NOT_AVAILABLE and
UNKNOWN. UNKNOWN must never be presented as confirmed availability.
