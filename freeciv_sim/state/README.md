# State and Movement

Role: encode Freeciv maps and entities, calculate movement, and supply simulator/live observations.

## Current state

- `multihead_state.py` defines the model-facing representation.
- `movement.py` handles map movement rules.
- `providers.py` contains local and Lua-backed state providers.

## TODO

- [ ] Implement or remove the unfinished `LuaRemoteProvider.resample()` path.
- [ ] Freeze an observation-schema version and provide checkpoint migration checks.
- [ ] Add property tests for action encoding/decoding, hex neighbors, wrapping, and map edges.
- [ ] Reduce duplicated normalization between simulator and live providers.
