# Freeciv Learning Environment for MuZero
This repo is Freeciv Learning Environment for MuZero based on [MuZero General](https://github.com/werner-duvaud/muzero-general)

## License

This project is distributed under the GNU General Public License version 3
(GPL-3.0-only).
Portions are derived from MuZero General, originally licensed under the MIT
License by Werner Duvaud. See [NOTICE](NOTICE) and [LICENSES/MIT.txt](LICENSES/MIT.txt).

## Layout

`freeciv_muzero` now carries its own Freeciv remote helpers and no longer depends on sibling `freeciv_alpha_zero` or `freeciv_rl` directories for training scripts.

Runtime assumptions that still remain outside this directory:

- A Freeciv build containing `freeciv-server`, `freeciv-gtk3.22`, and `run.sh`
- Freeciv data/scenarios if you have not copied them under this repository yet

## Setup

```bash
cd /home/hirokiokabe/freeciv_test/freeciv_muzero
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

This requirements file pins PyTorch to the official CUDA 12.1 wheels so it works with NVIDIA driver stacks that expose CUDA 12.2-class runtimes.

## Train
Example:
```bash
FREECIV_NO_SEA_UNITS=1 python3 muzero.py freeciv '{
"training_steps": 50000,
"num_simulations": 50,
"max_turns": 300,
"max_actions_per_turn": 50
}'
```

Headless live training:

```bash
cd /home/hirokiokabe/freeciv_test/freeciv_muzero
source .venv/bin/activate
./scripts/train.sh
```

To switch the Freeciv model trunk from square `3x3` convs to native hex-neighbor
convs, set `FREECIV_HEX_CONV=1`, for example
`FREECIV_HEX_CONV=1 ./scripts/train.sh`.

To force CPU mode, use `USE_GPU=0 ./scripts/train.sh`.
To target a specific GPU on a shared machine, set `CUDA_VISIBLE_DEVICES`, for example `CUDA_VISIBLE_DEVICES=5 ./scripts/train.sh`.
For the training wrappers, `TRAIN_GPU_LIST` is the easier form; it sets
`CUDA_VISIBLE_DEVICES` and, when `MUZERO_MAX_NUM_GPUS` is unset, matches Ray's
GPU count to the list length.

```bash
TRAIN_GPU_LIST=1,2,3,4,5 ./scripts/train_headless.sh
```

With `TRAIN_GPU_LIST`, training is pinned to the first GPU and self-play workers
are pinned round-robin to the remaining GPUs. Override the split explicitly with
`MUZERO_TRAIN_GPU_ID=1 MUZERO_SELFPLAY_GPU_IDS=2,3,4,5`.

Monitor GPU use with:

```bash
watch -n 1 nvidia-smi -i 1,2,3,4,5
nvidia-smi dmon -i 1,2,3,4,5 -s pucm
nvidia-smi pmon -i 1,2,3,4,5
```

If a previous crashed run left Ray workers behind, add `TRAIN_RESET_RAY=1`.
This stops the local Ray runtime before starting the new training run.

For multi-GPU headless training, increase the Ray self-play workers and expose
the GPUs:

```bash
TRAIN_GPU_LIST=1,2,3,4,5 \
NUM_WORKERS=4 \
TRAINING_STEPS=50000 \
NUM_SIMULATIONS=16 \
./scripts/train_headless.sh
```

Each worker gets its own Freeciv server and LuaRemote port by offsetting
`SERVER_PORT` and `LUA_PORT`. Override `FREECIV_SERVER_PORT_STRIDE` or
`FREECIV_LUAREMOTE_PORT_STRIDE` if a host has nearby occupied ports.

The default server rc used by the training scripts is [`start_generated_32x32.serv`](/home/hirokiokabe/freeciv_test/freeciv_muzero/start_generated_32x32.serv), which starts a 32x32 generated map. Set `FREECIV_GENERATED_MAP=0` to use the scenario path instead.

Action-space curriculum keeps the MuZero policy output size fixed and masks
legal actions by group during training:

```bash
FREECIV_ACTION_CURRICULUM_STAGE=0 ./scripts/train_headless.sh  # move/pass
FREECIV_ACTION_CURRICULUM_STAGE=1 ./scripts/train_headless.sh  # + build_city
FREECIV_ACTION_CURRICULUM_STAGE=2 ./scripts/train_headless.sh  # + research
FREECIV_ACTION_CURRICULUM_STAGE=3 ./scripts/train_headless.sh  # + production
FREECIV_ACTION_CURRICULUM_STAGE=4 ./scripts/train_headless.sh  # + attack
FREECIV_ACTION_CURRICULUM_STAGE=full ./scripts/train_headless.sh
```

Set `FREECIV_PRODUCTION_ESTIMATES=1` to query Freeciv's city shield stock,
shield surplus, effective build cost, and turns-to-completion through the
client Lua API. This populates the existing production-progress observation
and city production metadata. Leave it disabled when evaluating older
checkpoints that were trained without live production estimates.

Use `FREECIV_ACTION_CURRICULUM_GROUPS=move,build_city,research` for an explicit
group set. The current remote training environment still controls one MuZero
player in a Freeciv server; built-in AIs remain the external benchmark unless
`FREECIV_AIFILL` is overridden in the server rc template.
Run all stages sequentially with `./scripts/train_action_curriculum.sh`; override
`CURRICULUM_STAGES` and `CURRICULUM_TARGET_STEPS` for longer jobs.
The runner registers checkpoint symlinks in `results/model_registry`:
`latest.checkpoint`, `stage/<stage>.checkpoint`, and
`ladder/step-<target>.checkpoint`. Use the ladder checkpoints as fixed
opponents/evaluation anchors instead of evaluating only latest-vs-latest.

The training scripts now prefer repo-local paths first:

- `./freeciv_build_v3_2_uv`
- `./freeciv_build_v3_2`
- `../freeciv_build_v3_2_uv`
- `../freeciv_build_v3_2`

For scenarios they prefer:

- `./freeciv/data/scenarios/minimal_v4.sav`
- `./freeciv/scenarios/minimal_v4.sav`
- `~/.freeciv/scenarios/minimal_v4.sav`

Override either with `BUILD_DIR=...` or `SCENARIO_PATH=...` if needed.

Training and evaluation wrappers print elapsed time at the end and save logs
under `results/logs/` by default. Set `RUN_LOG=/path/to/run.log` to choose the
log path, or `SAVE_RUN_LOG=0` to disable wrapper logging.

## Gmail completion notification

Training and evaluation wrappers send a completion email when
`NOTIFY_EMAIL_TO` is set. The message includes success/failure, elapsed time,
host, result directory, checkpoint, and log path. Sending failure is reported
to stderr but does not change the run's exit status.

Install and configure `msmtp` once on each machine:

```bash
sudo apt install msmtp msmtp-mta ca-certificates
```

Example `~/.msmtprc` (use a Gmail app password, not the normal account
password). Gmail displays app passwords in four-character groups; remove the
spaces and enter the resulting 16 characters. Never commit the real password
to this repository:

```text
defaults
auth on
tls on
tls_starttls on
tls_trust_file /etc/ssl/certs/ca-certificates.crt

account gmail
host smtp.gmail.com
port 587
from sender@gmail.com
user sender@gmail.com
password YOUR_GMAIL_APP_PASSWORD

account default : gmail
```

Protect the configuration after creating it:

```bash
chmod 600 ~/.msmtprc
```

Send a test, then train:

```bash
NOTIFY_EMAIL_TO=destination@example.com ./scripts/send_test_email.sh
NOTIFY_EMAIL_TO=destination@example.com ./scripts/train_headless.sh
```

Optional variables: `NOTIFY_EMAIL_FROM`, `NOTIFY_EMAIL_SUBJECT_PREFIX`,
`NOTIFY_EMAIL_ON_SUCCESS=0`, and `NOTIFY_EMAIL_ON_FAILURE=0`.

## Google Drive sync

Set `GOOGLE_DRIVE_RESULTS` to mirror `results/` while training or evaluating.
Use an rclone remote for Google Drive:

```bash
GOOGLE_DRIVE_RESULTS=gdrive:freeciv_muzero/results ./scripts/train_headless.sh
GOOGLE_DRIVE_RESULTS=gdrive:freeciv_muzero/results ./scripts/eval_record_dual_view.sh
```

For a locally mounted Drive folder, pass the mount path instead:

```bash
GOOGLE_DRIVE_RESULTS="$HOME/Google Drive/freeciv_muzero/results" ./scripts/eval.sh
```

This repo also includes a small rclone mount helper:

```bash
rclone config create gdrive drive config_is_local=false
./scripts/mount_google_drive.sh
GOOGLE_DRIVE_RESULTS="$HOME/gdrive/freeciv_muzero/results" ./scripts/train_headless.sh
```

The sync runs in the background by default and is rate-limited to once every 300
seconds. Override with `GOOGLE_DRIVE_RESULTS_INTERVAL=60`, or set
`GOOGLE_DRIVE_RESULTS_BACKGROUND=0` to wait for each sync.

Optional strategic reward shaping:

```bash
FREECIV_REWARD_POTENTIAL=0.1 ./scripts/train.sh
```

This adds a potential-based term computed from cities, population, land,
military strength, research unlocks, production pipeline, exploration, and
city safety. To log reward components beside belief heatmaps, set
`FREECIV_REWARD_TENSORBOARD=1` or enable `FREECIV_BELIEF_TENSORBOARD=1`.
To append the belief tracker planes to the model observation, set
`FREECIV_OBSERVE_BELIEF=1`.

## Test
Example:
```bash
python3 remote_play.py \
    --checkpoint path/to/checkpoint/model.checkpoint \
    --map-width 32 --map-height 32 --max-turns 300 \
    --max-actions-per-turn 100 \
    --num-simulations 50 --temperature 0.0 \
    --no-sea-units \
    --host 127.0.0.1 --port 4444 \
    --player-id 0
```

GUI evaluation wrapper:
```bash
cd /home/hirokiokabe/freeciv_test/freeciv_muzero
source .venv/bin/activate
./scripts/eval.sh
```

Hex-neighbor wiring sanity check:

```bash
.venv/bin/python scripts/check_hex_conv.py
```

It picks the latest `model.checkpoint` from `results/checkpoints` or `results/freeciv_remote` by default.
Override it with `CHECKPOINT=... ./scripts/eval.sh` or `./scripts/eval.sh /path/to/model.checkpoint`.

Record gameplay with Freeciv-style hex heatmap panels:

```bash
./scripts/eval_record_with_heatmaps.sh
```

Set `HEATMAP_TILE_SHAPE=square` to render the previous square-grid panels.
The recommended recorder is `eval_record_dual_view.sh`. It runs one evaluation
game and keeps separate `eval-agent.mp4`, `eval-global.mp4`, map-only videos,
and heatmap videos:

```bash
./scripts/eval_record_dual_view.sh
```

For multiple recorded evaluation games, use `eval_record_dual_parallel.sh`.
It is a batch wrapper around `eval_record_dual_view.sh` and writes one
`game-XX/` directory per game.

See [`scripts/README.md`](scripts/README.md) for the script map.

Recordings are ignored by default. To publish selected MP4 recordings through Git LFS:

```bash
scripts/stage_recording_lfs.sh results/evals/YYYYmmdd-HHMMSS/eval-agent.mp4
git commit -m "Add evaluation recording"
git push
```

## Results

Training checkpoints, replay buffers, evaluation videos, and other generated
artifacts under `results/` can be stored in Google Drive with
`GOOGLE_DRIVE_RESULTS`, as described above. Google Drive is the preferred
long-term store for these large files; they do not need to be committed to Git
or Git LFS after a successful sync. Keep only small summaries or deliberately
selected artifacts in the repository.

Example:

https://github.com/user-attachments/assets/02942ee1-d173-4deb-b7ed-cb7ad01f3233
