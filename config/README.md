# Freeciv Configuration Snapshots

Role: provide simulator-readable snapshots of buildings, research, rules, terrain, and units.

## Current state

- JSON files are consumed by the local Freeciv simulator.
- Their source Freeciv revision and export procedure are not encoded in the files.

## TODO

- [ ] Add schema version, Freeciv commit, and ruleset hash metadata.
- [ ] Generate snapshots reproducibly with `scripts/export_ruleset_config.py`.
- [ ] Add a drift check comparing exported data with committed snapshots.
- [ ] Validate identifiers against the live Lua API before training.
