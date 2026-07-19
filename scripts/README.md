# Scripts

Direct entrypoints:

- `train_headless.sh`: Main training entrypoint. Runs Xvfb, Freeciv server/client, and Ray MuZero training.
- `eval_record_dual_view.sh`: Main recorded evaluation entrypoint. Produces one game with agent view, global observer view, map-only videos, and heatmap videos.

Required dependencies:

- `common.sh`: Shared shell helpers used by `train_headless.sh` and `eval_record_dual_view.sh`.
- `eval_headless.sh`: Evaluation runner called by `eval_record_dual_view.sh`.
- `render_tb_heatmap_panel.py`: Heatmap frame renderer called by `eval_record_dual_view.sh`.
- `filter_freeciv_eval_log.awk`: Log filter used by `eval_headless.sh`.

Minimum keep set:

- `train_headless.sh`
- `eval_record_dual_view.sh`
- `eval_headless.sh`
- `common.sh`
- `render_tb_heatmap_panel.py`
- `filter_freeciv_eval_log.awk`

Archive candidates:

- `train.sh`: Short alias for `train_headless.sh`.
- `eval.sh`: Short alias for `eval_headless.sh`.
- `eval_gui.sh`: Evaluation with GUI display.
- `eval_record.sh`: Older single-view recorder.
- `eval_record_with_heatmaps.sh`: Older single-view recorder plus heatmap generation.
- `eval_record_dual_parallel.sh`: Parallel multi-game recorder. Can be archived if unused.
- `eval_headless_dual_agents.sh`: Two-checkpoint / two-agent comparison.
- `train_gui.sh`: Training with GUI display.
- `train_action_curriculum.sh`: Staged action-curriculum training.
- `run_mcts_ablation.sh`: Trains and tests the four Wasserstein/stochastic MCTS combinations.
- `register_model.sh`: Checkpoint registry helper.
- `inspect_checkpoint.sh`: Checkpoint inspection helper.
- `freeciv_headless.sh`: Standalone Freeciv headless runner.
- `freeciv_gui.sh`: Standalone Freeciv GUI runner.
- `freeciv_kill_clients.sh`: Freeciv client cleanup helper.
- `render_threat_overlay_video.sh`: Threat overlay video renderer.
- `export_tb_heatmap_overlay.py`: Heatmap overlay exporter.
- `summarize_eval_run.py`: Evaluation run summarizer.
- `stage_recording_lfs.sh`: Git LFS staging helper for selected MP4 recordings.
- `check_hex_conv.py`: Hex-convolution sanity check.

`eval_record_dual_view.sh` vs `eval_record_dual_parallel.sh`:

- `dual_view`: Runs one game. Output goes to `results/evals/<RUN_STAMP>/`.
- `dual_parallel`: Runs N games in parallel by launching `dual_view` for each game. Output goes to `results/evals/<RUN_STAMP>/game-XX/`.
- `dual_view`: Use for normal recording checks, video generation, and debugging.
- `dual_parallel`: Use for multi-run win-rate or stability evaluation across multiple GPUs.

`eval_record_dual_view.sh` lets `remote_play.py` send `/take` and `/start` by
default. Keep `RECORD_EXTERNAL_START=0` unless you are also changing the
Freeciv control-state startup path.

Common examples:

```bash
# train
./scripts/train_headless.sh

# short stochastic smoke train
MUZERO_STOCHASTIC=1 USE_GPU=0 TRAINING_STEPS=1 MUZERO_CHECKPOINT_INTERVAL=1 \
NUM_SIMULATIONS=1 MAX_TURNS=1 ./scripts/train_headless.sh

# four-condition MCTS ablation; outputs summary.tsv under results/mcts_ablation
TRAINING_STEPS=10000 NUM_TESTS=10 ABLATION_SEEDS=1,2,3 \
  ./scripts/run_mcts_ablation.sh

# record one evaluation
./scripts/eval_record_dual_view.sh results/freeciv_remote/.../model.checkpoint

# optional: record 20 evaluations in parallel
GAMES=20 GPU_LIST=0,1,2,3,4 ./scripts/eval_record_dual_parallel.sh \
  results/freeciv_remote/.../model.checkpoint
```

The ablation runner defaults to `gdrive:freeciv_muzero/results`. Checkpoints,
replay buffers, and TensorBoard event files are synced together under the same
relative experiment directory. Override `GOOGLE_DRIVE_RESULTS` to use another
remote or mounted Drive path.
