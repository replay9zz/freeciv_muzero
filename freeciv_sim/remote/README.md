# Live Freeciv Remote Control

Role: communicate with a running Freeciv client through Lua and expose live games to MuZero.

## Current state

- `lua_client.py` handles RPC transport.
- `lua_queries.py` and `lua_actions.py` define the protocol surface.
- `session.py` manages startup, player discovery, reset, and shutdown.

## TODO

- [x] Implement reliable terminal/winner detection from live server state.
- [ ] Batch snapshot queries and profile per-turn RPC latency.
- [ ] Test reconnect, timeout, partial reply, server exit, and reset failures.
- [ ] Add an opt-in smoke test against a locally running Freeciv stack.
- [ ] Version and validate the Lua/Python protocol at session startup.
