# Tests

Role: protect simulator economics, remote-session behavior, and outcome parsing.

## Current state

- Tests use Python's `unittest` discovery.
- Live Freeciv is not required by the default suite.

## TODO

- [ ] Cover action masks, state transitions, rewards, and simulator/live parity fixtures.
- [ ] Test malformed Lua replies, reconnects, timeouts, and server process failures.
- [ ] Add an opt-in live smoke suite with explicit ports and cleanup guarantees.
- [ ] Add dataset/checkpoint schema compatibility tests.
