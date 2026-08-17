#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed. Run: .venv/bin/python -m pip install 'optuna>=4,<5'"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freeciv_sim.evaluation.outcome import GameOutcome, optuna_objective

RESULT_RE = re.compile(
    r"\[selfplay-result\]\s+"
    r"outcome=(?P<outcome>-?\d+(?:\.\d+)?)\s+"
    r"win_point=(?P<win_point>\d+(?:\.\d+)?)\s+"
    r"own_score=(?P<own_score>-?\d+(?:\.\d+)?)?\s+"
    r"opponent_score=(?P<opponent_score>-?\d+(?:\.\d+)?)?\s+"
    r"decided_by=(?P<decided_by>\w+)"
)

BASELINE = {
    "batch_size": 64,
    "replay_buffer_size": 10,
    "num_unroll_steps": 10,
    "lr_init": 0.01,
    "wasserstein_uncertainty_coef": 0.25,
    "wonder_min_turn": 0,
    "wonder_min_cities": 1,
}


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be comma-separated non-negative integers")
    return seeds


def parse_result(path: Path) -> GameOutcome:
    matches = list(RESULT_RE.finditer(path.read_text(errors="replace")))
    if not matches:
        raise RuntimeError(f"selfplay result not found in {path}")
    values = matches[-1].groupdict()

    def optional_float(value: str | None) -> float | None:
        return None if value in (None, "") else float(value)

    return GameOutcome(
        value=float(values["outcome"]),
        win_point=float(values["win_point"]),
        own_score=optional_float(values["own_score"]),
        opponent_score=optional_float(values["opponent_score"]),
        decided_by=values["decided_by"],
    )


def run_checked(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit status {result.returncode}: {command[0]}")


def write_trials(study: optuna.Study, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    param_names = sorted({name for trial in study.trials for name in trial.params})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "state", "value", "win_rate", *param_names])
        for trial in study.trials:
            writer.writerow(
                [
                    trial.number,
                    trial.state.name,
                    "" if trial.value is None else trial.value,
                    trial.user_attrs.get("win_rate", ""),
                    *(trial.params.get(name, "") for name in param_names),
                ]
            )


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(
        description="Tune MuZero learning settings using held-out game outcomes, never shaped rewards."
    )
    out.add_argument("--study-name", default="freeciv-outcomes")
    out.add_argument("--output", type=Path, default=ROOT / "results" / "outcome_optuna")
    out.add_argument("--storage")
    out.add_argument("--trials", type=int, default=20)
    out.add_argument("--timeout", type=int)
    out.add_argument("--train-seeds", type=parse_seeds, default=parse_seeds("0,1,2"))
    out.add_argument("--eval-seeds", type=parse_seeds, default=parse_seeds("100,101,102,103,104"))
    out.add_argument("--training-steps", type=int, default=5000)
    out.add_argument("--max-turns", type=int, default=300)
    out.add_argument("--train-simulations", type=int, default=2)
    out.add_argument("--test-simulations", type=int, default=16)
    out.add_argument("--num-workers", type=int, default=1)
    out.add_argument("--imitation-checkpoint", type=Path)
    out.add_argument(
        "--wonder-policy-only",
        action="store_true",
        help="Tune only early great-wonder unlock thresholds; keep learning settings fixed.",
    )
    out.add_argument("--keep-replay-buffers", action="store_true")
    out.add_argument("--check", action="store_true")
    return out


def main() -> int:
    args = parser().parse_args()
    if args.trials < 1 or args.training_steps < 1 or args.max_turns < 1:
        raise SystemExit("trials, training-steps, and max-turns must be positive")
    if set(args.train_seeds) & set(args.eval_seeds):
        raise SystemExit("training and evaluation seeds must be disjoint")
    imitation_checkpoint = args.imitation_checkpoint.resolve() if args.imitation_checkpoint else None
    if imitation_checkpoint is not None and not imitation_checkpoint.is_file():
        raise SystemExit(f"imitation checkpoint not found: {imitation_checkpoint}")

    output = args.output.resolve() / args.study_name
    output.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{output / 'study.db'}"
    config = {
        "train_seeds": args.train_seeds,
        "eval_seeds": args.eval_seeds,
        "training_steps": args.training_steps,
        "max_turns": args.max_turns,
        "train_simulations": args.train_simulations,
        "test_simulations": args.test_simulations,
        "num_workers": args.num_workers,
        "imitation_checkpoint": str(imitation_checkpoint or ""),
        "objective": "win_rate_then_normalized_score_margin",
        "map": [32, 32],
        "mcts_backup_operator": "wasserstein",
        "wonder_policy_only": args.wonder_policy_only,
    }
    if args.check:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=20260801),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )
    saved = study.user_attrs.get("objective_config")
    if saved is not None and saved != config:
        raise SystemExit("Study settings changed; use a new --study-name.")
    if saved is None:
        study.set_user_attr("objective_config", config)
        study.enqueue_trial(
            {
                "wonder_min_turn": 0,
                "wonder_min_cities": 1,
                **({} if args.wonder_policy_only else {
                    key: value
                    for key, value in BASELINE.items()
                    if not key.startswith("wonder_")
                }),
            }
        )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "wonder_min_turn": trial.suggest_categorical(
                "wonder_min_turn", [0, 30, 60, 90]
            ),
            "wonder_min_cities": trial.suggest_categorical(
                "wonder_min_cities", [1, 2, 3, 4]
            ),
        }
        if args.wonder_policy_only:
            params.update(
                batch_size=64,
                replay_buffer_size=10,
                num_unroll_steps=10,
                lr_init=0.001,
                wasserstein_uncertainty_coef=0.25,
            )
        else:
            params.update(
                batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                replay_buffer_size=trial.suggest_categorical(
                    "replay_buffer_size", [4, 8, 10]
                ),
                num_unroll_steps=trial.suggest_categorical(
                    "num_unroll_steps", [5, 10, 20]
                ),
                lr_init=trial.suggest_float("lr_init", 1e-4, 2e-2, log=True),
                wasserstein_uncertainty_coef=trial.suggest_float(
                    "wasserstein_uncertainty_coef", 0.0, 0.5
                ),
            )
        trial_dir = output / f"trial-{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "params.json").write_text(
            json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        outcomes: list[GameOutcome] = []

        for train_index, train_seed in enumerate(args.train_seeds):
            train_dir = trial_dir / f"train-seed-{train_seed}"
            env = os.environ.copy()
            env.update(
                {
                    "MUZERO_RESULTS_PATH": str(train_dir),
                    "MUZERO_SEED": str(train_seed),
                    "FREECIV_SEED": str(train_seed),
                    "CHECKPOINT_PATH": str(imitation_checkpoint or ""),
                    "TRAINING_STEPS": str(args.training_steps),
                    "MAX_TURNS": str(args.max_turns),
                    "NUM_SIMULATIONS": str(args.train_simulations),
                    "NUM_WORKERS": str(args.num_workers),
                    "MUZERO_BATCH_SIZE": str(params["batch_size"]),
                    "MUZERO_REPLAY_BUFFER_SIZE": str(params["replay_buffer_size"]),
                    "MUZERO_NUM_UNROLL_STEPS": str(params["num_unroll_steps"]),
                    "MUZERO_LR_INIT": str(params["lr_init"]),
                    "MUZERO_MCTS_BACKUP_OPERATOR": "wasserstein",
                    "MUZERO_MCTS_WASSERSTEIN_UNCERTAINTY_COEF": str(
                        params["wasserstein_uncertainty_coef"]
                    ),
                    "FREECIV_WONDER_MIN_TURN": str(params["wonder_min_turn"]),
                    "FREECIV_WONDER_MIN_CITIES": str(params["wonder_min_cities"]),
                    "FREECIV_PRODUCTION_ESTIMATES": "1",
                    "GOOGLE_DRIVE_RESULTS": "",
                    "NOTIFY_EMAIL_TO": "",
                }
            )
            replay_path = train_dir / "replay_buffer.pkl"
            run_checked([str(ROOT / "scripts" / "train_headless.sh")], env)
            checkpoint = train_dir / "model.checkpoint"
            if not checkpoint.is_file():
                raise RuntimeError(f"checkpoint missing: {checkpoint}")

            for eval_seed in args.eval_seeds:
                eval_dir = train_dir / "eval"
                eval_dir.mkdir(parents=True, exist_ok=True)
                eval_log = eval_dir / f"seed-{eval_seed}.log"
                test_env = env.copy()
                test_env.update(
                    {
                        "CHECKPOINT_PATH": str(checkpoint),
                        "TEST_LOG_PATH": str(eval_log),
                        "FREECIV_SEED": str(eval_seed),
                        "MUZERO_SEED": str(eval_seed),
                        "FREECIV_SCORE_RUN_ID": (
                            f"{args.study_name}-trial{trial.number}-train{train_seed}-eval{eval_seed}"
                        ),
                        "NUM_TESTS": "1",
                        "NUM_SIMULATIONS": str(args.test_simulations),
                        "MAX_TURNS": str(args.max_turns),
                    }
                )
                run_checked([str(ROOT / "scripts" / "test_headless.sh")], test_env)
                outcomes.append(parse_result(eval_log))

            running = optuna_objective(outcomes)
            trial.report(running, train_index)
            if trial.should_prune():
                raise optuna.TrialPruned(f"objective={running:.6f}")
            if replay_path.is_file() and not args.keep_replay_buffers:
                replay_path.unlink()

        value = optuna_objective(outcomes)
        trial.set_user_attr("win_rate", statistics.fmean(item.win_point for item in outcomes))
        trial.set_user_attr("outcomes", [item.__dict__ for item in outcomes])
        return value

    counted_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }
    finished = sum(trial.state in counted_states for trial in study.trials)
    try:
        if finished < args.trials:
            study.optimize(
                objective,
                n_trials=args.trials - finished,
                timeout=args.timeout,
                catch=(RuntimeError,),
            )
    finally:
        write_trials(study, output / "trials.csv")
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            best = study.best_trial
            summary = {
                "study": study.study_name,
                "best_trial": best.number,
                "best_value": best.value,
                "win_rate": best.user_attrs.get("win_rate"),
                "best_params": best.params,
            }
            (output / "best.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
