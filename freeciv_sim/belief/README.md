# Belief Tracking

Role: estimate hidden enemy locations and threats outside current visibility.

## Current state

- `tracker.py` uses heuristic diffusion and threat updates.

## TODO

- [ ] Calibrate belief probabilities against hidden-state ground truth from recorded games.
- [ ] Include unit speed, terrain cost, transports, and map wrapping in diffusion.
- [ ] Test visibility clearing, stale contacts, city observations, and edge tiles.
- [ ] Expose belief-quality metrics for evaluation and ablation.
