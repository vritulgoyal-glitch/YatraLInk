"""Development railway-data providers (Phase 2B).

This package holds adapter-style providers that map a data source into the
Phase 2A normalized contract (:mod:`engine.railway_data`).  The only provider
shipped here is the **development timetable fixture** provider
(:mod:`engine.railway_data.providers.timetable_fixture`), which proves the data
path::

    railway timetable data
        -> provider adapter (this package)
        -> Phase 2A normalized models
        -> Phase 1 journey engine

It contains no network access, no API keys, no scraping, no booking logic and
no live availability of any kind.  A future *authorised* production provider
will replace the fixture adapter behind the same Phase 2A models.
"""

from engine.railway_data.providers.timetable_fixture import TimetableFixtureProvider

__all__ = ["TimetableFixtureProvider"]
