# Freeciv Integration Package

Role: connect MuZero to simulated and live Freeciv state, rules, agents, and evaluation.

## Current state

- Subpackages separate agents, belief, evaluation, imitation, remote control, rules, and state.
- Public interfaces are mostly inferred from their consumers.

## TODO

- [ ] Define stable typed interfaces between game adapters and these subpackages.
- [ ] Document package ownership and dependency direction to prevent circular imports.
- [ ] Add package-level simulator/live integration tests.
