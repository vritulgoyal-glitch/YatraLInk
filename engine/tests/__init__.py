"""YatraLink engine test suite.

Tests are organised by concern so that a failure points at one subsystem:

* ``test_time``             day offsets, midnight crossing, duration arithmetic
* ``test_models``           domain model invariants and station normalisation
* ``test_search_request``   request validation
* ``test_fixtures``         fixture loading and the synthetic snapshot's shape
* ``test_direct``           direct journey generation
* ``test_same_train_split`` same-train split generation
* ``test_connections``      transfer validation and risk classification
* ``test_invalid_journeys`` rejection paths and impossible candidates
* ``test_availability``     availability states, aggregation and UNKNOWN safety
* ``test_ranking``          scoring, determinism and tiebreakers
* ``test_pipeline``         the end-to-end pipeline, deduplication and limits
"""
