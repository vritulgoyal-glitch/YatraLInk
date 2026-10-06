# YatraLink Connection Engine

This directory contains the deterministic railway journey optimization engine.

Planned responsibilities:

- Direct journey detection
- Same-train split journey detection
- Connecting-train candidate generation
- Connection validation
- Transfer-time calculation
- Journey duration calculation
- Fare aggregation
- Availability-state handling
- Connection-risk calculation
- Candidate deduplication
- Journey ranking

The engine must not invent railway facts. It operates on validated railway
data supplied by the data/service layers.
