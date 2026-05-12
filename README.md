# Freeciv Learning Environment for MuZero
This repo is Freeciv Learning Environment for MuZero based on [MuZero General](https://github.com/werner-duvaud/muzero-general)

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
