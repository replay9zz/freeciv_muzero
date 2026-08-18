# Game Adapters

Role: expose supported environments through the common MuZero game interface.

## Current state

- Generic example games coexist with `freeciv`, `freeciv_remote`, and `tech_policy` adapters.
- `freeciv_remote.py` currently owns several live-session responsibilities.

## TODO

- [ ] Split transport/session orchestration from Freeciv game and reward logic.
- [ ] Enforce simulator/live parity for observations, actions, masks, and rewards.
- [ ] Define multiplayer semantics; generic self-play currently has incomplete support beyond two players.
- [ ] Add fast import/reset/step smoke tests for maintained adapters.
- [ ] Mark inherited example games as supported, reference-only, or removable.
