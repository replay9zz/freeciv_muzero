# Freeciv Learning Environment for MuZero
This repo is Freeciv Learning Environment for MuZero based on [MuZero General](https://github.com/werner-duvaud/muzero-general)

## Layout

`freeciv_muzero` now carries its own Freeciv remote helpers and no longer depends on sibling `freeciv_alpha_zero` or `freeciv_rl` directories.

Runtime assumptions that still remain outside this directory:

- `../freeciv_build_v3_2_uv`: Freeciv build with `freeciv-server`, `freeciv-gtk3.22`, and `run.sh`
- `../freeciv`: data directory used for rulesets and scenarios

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
./train_headless.sh
```

To force CPU mode, use `USE_GPU=0 ./train_headless.sh`.
To target a specific GPU on a shared machine, set `CUDA_VISIBLE_DEVICES`, for example `CUDA_VISIBLE_DEVICES=5 ./train_headless.sh`.

The default server rc used by the training scripts is [`start_single.serv`](/home/hirokiokabe/freeciv_test/freeciv_muzero/start_single.serv).
The default build directory is `../freeciv_build_v3_2_uv`; override with `BUILD_DIR=...` if needed.

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

## Results
Example:

https://github.com/user-attachments/assets/02942ee1-d173-4deb-b7ed-cb7ad01f3233
