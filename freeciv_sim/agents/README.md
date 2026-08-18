# Agents

Role: produce exploration, combat, production, and research decisions and route them by role.

## Current state

- Specialized heuristic agents share a lightweight router.

## TODO

- [ ] Define arbitration when multiple agents propose actions for the same entity or turn.
- [ ] Test every role against legal-action masks and empty candidate sets.
- [ ] Measure each agent's contribution with reproducible ablations.
- [ ] Record the boundary between heuristic decisions and MuZero decisions.
