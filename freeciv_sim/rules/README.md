# Rules

Role: load Freeciv ruleset snapshots and model research dependencies.

## Current state

- `ruleset_loader.py` reads exported configuration.
- `research.py` provides technology-tree behavior.

## TODO

- [ ] Cover building effects, unit obsolescence, governments, and ruleset effects used by decisions.
- [ ] Reject incompatible snapshot schema/ruleset hashes early.
- [ ] Add parity tests against representative live Lua queries.
- [ ] Document fallback behavior for missing or custom ruleset fields.
