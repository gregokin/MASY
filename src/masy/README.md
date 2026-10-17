# MASY wrapper namespace

MASY-only code belongs here; no upstream wind/water physics is copied. As of MASY-01 the only executable code is the read-only current-source compatibility preflight (`config`, `sourceobs`, `importcheck`, `contracts`, `clock`, `report`, `cli`), started by `tools/masy_preflight.py` (see `docs/preflight.md`). It observes, imports under guards and reports. It never runs transport and never reports a pair as ready.

Not implemented yet: schedule, alternating wind/water transport, native state/bed adapters, accounting, combined restart, immutable runtime, and performance or GPU work. API passes are interface evidence only and qualify no physics, restart or alternating event.
