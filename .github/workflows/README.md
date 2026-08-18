# GitHub Workflows

Role: automated checks for the GitHub mirror.

## Current state

- `ci-testing.yaml` runs the inherited MuZero test workflow.
- It uses an old Python/runtime and long CartPole training as its main check.

## TODO

- [ ] Upgrade pinned actions and test on the Python version used by this project.
- [ ] Run `python -m unittest discover -s tests` and shell syntax checks on every change.
- [ ] Split fast pull-request checks from scheduled long training/evaluation jobs.
- [ ] Cache dependencies and retain compact failure logs instead of large generated artifacts.
