# Imitation Learning

Role: collect built-in AI trajectories, build datasets, and pretrain a policy.

## Current state

- Collection, dataset construction, and training are separate command-line stages.

## TODO

- [ ] Version dataset, action, observation, ruleset, and checkpoint schemas together.
- [ ] Prevent game/seed leakage between training and validation splits.
- [ ] Validate recorded actions against the legal-action mask before training.
- [ ] Save a reproducibility manifest with each dataset and pretrained checkpoint.
