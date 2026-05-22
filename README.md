# Freeciv Learning Environment for MuZero
This repo is Freeciv Learning Environment for MuZero based on [MuZero General](https://github.com/werner-duvaud/muzero-general)

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
For multi-GPU headless training, increase the Ray self-play workers and expose
the GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 \
MUZERO_MAX_NUM_GPUS=10 \
NUM_WORKERS=8 \
TRAINING_STEPS=50000 \
NUM_SIMULATIONS=16 \
./scripts/train_headless.sh
```

Each worker gets its own Freeciv server and LuaRemote port by offsetting
`SERVER_PORT` and `LUA_PORT`. Override `FREECIV_SERVER_PORT_STRIDE` or
`FREECIV_LUAREMOTE_PORT_STRIDE` if a host has nearby occupied ports.

The default server rc used by the training scripts is [`start_single.serv`](/home/hirokiokabe/freeciv_test/freeciv_muzero/start_single.serv).
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
    --map-width 4 --map-height 16 --max-turns 300 \
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

Recordings are ignored by default. To publish selected MP4 recordings through Git LFS:

```bash
scripts/stage_recording_lfs.sh results/evals/YYYYmmdd-HHMMSS/eval-agent.mp4
git commit -m "Add evaluation recording"
git push
```

## Results
Example:

https://github.com/user-attachments/assets/02942ee1-d173-4deb-b7ed-cb7ad01f3233
