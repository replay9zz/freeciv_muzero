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
except ImportError as exc:  # pragma: no cover - startup guidance
    raise SystemExit(
        "Optuna is not installed. Run: .venv/bin/python -m pip install 'optuna>=4,<5'"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
SELFPLAY_SCORE_RE = re.compile(
    r"\[selfplay\]\s+turn=(?P<turn>\d+).*?\bciv_score=(?P<score>-?\d+(?:\.\d+)?)"
)

REWARD_RANGES = {
    "FREECIV_REWARD_CITY": (2.0, 12.0),
    "FREECIV_REWARD_SETTLER": (0.0, 3.0),
    "FREECIV_REWARD_POPULATION": (0.5, 2.0),
    "FREECIV_REWARD_EXPLORE": (0.1, 1.0),
    "FREECIV_REWARD_CIV_SCORE": (0.25, 1.5),
    "FREECIV_REWARD_POTENTIAL": (0.0, 0.1),
}

BASELINE_REWARDS = {
    "FREECIV_REWARD_CITY": 12.0,
    "FREECIV_REWARD_SETTLER": 3.0,
    "FREECIV_REWARD_POPULATION": 2.0,
    "FREECIV_REWARD_EXPLORE": 1.0,
    "FREECIV_REWARD_CIV_SCORE": 0.25,
    "FREECIV_REWARD_POTENTIAL": 0.0,
}


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be comma-separated non-negative integers")
    return seeds


def parse_final_civ_scores(path: Path) -> list[float]:
    scores: list[float] = []
    last_turn: int | None = None
    last_score: float | None = None
    for line in path.read_text(errors="replace").splitlines():
        match = SELFPLAY_SCORE_RE.search(line)
        if not match:
            continue
        turn = int(match.group("turn"))
        score = float(match.group("score"))
        if last_turn is not None and turn < last_turn and last_score is not None:
            scores.append(last_score)
        last_turn = turn
        last_score = score
    if last_score is not None:
        scores.append(last_score)
    return scores


def write_trials_csv(study: optuna.Study, path: Path) -> None:
    param_names = sorted(REWARD_RANGES)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "state", "value", *param_names])
        for trial in study.trials:
            writer.writerow(
                [
                    trial.number,
                    trial.state.name,
                    "" if trial.value is None else trial.value,
                    *(trial.params.get(name, "") for name in param_names),
                ]
            )


def run_checked(command: list[str], env: dict[str, str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit status {result.returncode}: {command[0]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize Freeciv reward weights with Optuna and fixed-score evaluation."
    )
    parser.add_argument("--study-name", default="freeciv-rewards")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "reward_optuna")
    parser.add_argument("--storage", help="Optuna storage URL; defaults to output/study.db")
    parser.add_argument(
        "--trials", type=int, default=20, help="Target total trial count (default: 20)"
    )
    parser.add_argument("--timeout", type=int, help="Overall optimization timeout in seconds")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2"))
    parser.add_argument("--training-steps", type=int, default=5000)
    parser.add_argument("--train-max-turns", type=int, default=200)
    parser.add_argument("--test-max-turns", type=int, default=100)
    parser.add_argument("--num-tests", type=int, default=3)
    parser.add_argument("--train-simulations", type=int, default=2)
    parser.add_argument("--test-simulations", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampler-seed", type=int, default=20260722)
    parser.add_argument("--keep-replay-buffers", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate setup without training")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.trials < 1 or args.training_steps < 1 or args.num_tests < 1:
        raise SystemExit("trials, training-steps, and num-tests must be positive")

    output = args.output.resolve() / args.study_name
    output.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{output / 'study.db'}"

    if args.check:
        print(f"root: {ROOT}")
        print(f"output: {output}")
        print(f"storage: {storage}")
        print(f"seeds: {','.join(map(str, args.seeds))}")
        print("objective: mean final civ_score (independent of shaped reward)")
        print("status: ok")
        return 0

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    objective_config = {
        "seeds": args.seeds,
        "training_steps": args.training_steps,
        "train_max_turns": args.train_max_turns,
        "test_max_turns": args.test_max_turns,
        "num_tests": args.num_tests,
        "train_simulations": args.train_simulations,
        "test_simulations": args.test_simulations,
        "batch_size": args.batch_size,
        "sampler_seed": args.sampler_seed,
        "mcts_backup_operator": "wasserstein",
        "stochastic_muzero": False,
        "objective": "mean_final_civ_score",
        "reward_ranges": {
            name: [low, high] for name, (low, high) in REWARD_RANGES.items()
        },
    }
    saved_config = study.user_attrs.get("objective_config")
    if saved_config is not None and saved_config != objective_config:
        raise SystemExit(
            "Study settings differ from the existing study. "
            "Use a new --study-name, or restore the original settings.\n"
            f"saved: {json.dumps(saved_config, sort_keys=True)}\n"
            f"given: {json.dumps(objective_config, sort_keys=True)}"
        )
    if saved_config is None:
        study.set_user_attr("objective_config", objective_config)
    if not args.no_baseline and not study.user_attrs.get("baseline_enqueued"):
        study.enqueue_trial(BASELINE_REWARDS)
        study.set_user_attr("baseline_enqueued", True)

    def objective(trial: optuna.Trial) -> float:
        rewards = {
            name: trial.suggest_float(name, low, high)
            for name, (low, high) in REWARD_RANGES.items()
        }
        trial_dir = output / f"trial-{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "params.json").write_text(
            json.dumps(rewards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        all_scores: list[float] = []
        for seed_index, seed in enumerate(args.seeds):
            seed_dir = trial_dir / f"seed-{seed}"
            train_dir = seed_dir / "train"
            test_log = seed_dir / "test.log"
            seed_dir.mkdir(parents=True, exist_ok=True)
            run_key = f"{args.study_name}-trial{trial.number:04d}-seed{seed}"
            env = os.environ.copy()
            env.update({name: str(value) for name, value in rewards.items()})
            env.update(
                {
                    "MUZERO_RESULTS_PATH": str(train_dir),
                    "MUZERO_SEED": str(seed),
                    "FREECIV_SEED": str(seed),
                    "MUZERO_STOCHASTIC": "0",
                    "MUZERO_MCTS_BACKUP_OPERATOR": "wasserstein",
                    "FREECIV_PRODUCTION_ESTIMATES": "1",
                    "FREECIV_REWARD_TENSORBOARD": "1",
                    "FREECIV_BELIEF_TENSORBOARD_DIR": str(
                        seed_dir / "reward_tensorboard" / "train"
                    ),
                    "FREECIV_SCORE_RUN_ID": f"{run_key}-train",
                    "FREECIV_PROCESS_LOG": str(seed_dir / "freeciv-train.log"),
                    "RUN_LOG": str(seed_dir / "train.log"),
                    "TRAINING_STEPS": str(args.training_steps),
                    "MAX_TURNS": str(args.train_max_turns),
                    "NUM_SIMULATIONS": str(args.train_simulations),
                    "MUZERO_BATCH_SIZE": str(args.batch_size),
                    "MUZERO_CHECKPOINT_INTERVAL": str(min(10, args.training_steps)),
                    "GOOGLE_DRIVE_RESULTS": "",
                    "NOTIFY_EMAIL_TO": "",
                }
            )

            replay_path = train_dir / "replay_buffer.pkl"
            try:
                run_checked([str(ROOT / "scripts" / "train_headless.sh")], env, ROOT)
                checkpoint = train_dir / "model.checkpoint"
                if not checkpoint.is_file():
                    raise RuntimeError(f"checkpoint missing: {checkpoint}")

                test_env = env.copy()
                test_env.update(
                    {
                        "CHECKPOINT_PATH": str(checkpoint),
                        "TEST_LOG_PATH": str(test_log),
                        "FREECIV_SCORE_RUN_ID": f"{run_key}-test",
                        "FREECIV_PROCESS_LOG": str(seed_dir / "freeciv-test.log"),
                        "FREECIV_BELIEF_TENSORBOARD_DIR": str(
                            seed_dir / "reward_tensorboard" / "test"
                        ),
                        "NUM_TESTS": str(args.num_tests),
                        "MAX_TURNS": str(args.test_max_turns),
                        "NUM_SIMULATIONS": str(args.test_simulations),
                    }
                )
                run_checked([str(ROOT / "scripts" / "test_headless.sh")], test_env, ROOT)
                scores = parse_final_civ_scores(test_log)
                if len(scores) != args.num_tests:
                    raise RuntimeError(
                        f"expected {args.num_tests} final scores, found {len(scores)} in {test_log}"
                    )
                all_scores.extend(scores)
                running_mean = statistics.fmean(all_scores)
                trial.report(running_mean, seed_index)
                trial.set_user_attr(f"seed_{seed}_scores", scores)
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"mean civ_score={running_mean:.3f} after {seed_index + 1} seeds"
                    )
            except RuntimeError as exc:
                trial.set_user_attr(f"seed_{seed}_error", str(exc))
                raise
            finally:
                if replay_path.is_file() and not args.keep_replay_buffers:
                    replay_path.unlink()

        score = statistics.fmean(all_scores)
        trial.set_user_attr("scores", all_scores)
        trial.set_user_attr("score_stddev", statistics.pstdev(all_scores))
        return score

    finished_trials = sum(trial.state.is_finished() for trial in study.trials)
    remaining_trials = max(0, args.trials - finished_trials)
    try:
        if remaining_trials:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=args.timeout,
                catch=(RuntimeError,),
            )
        else:
            print(
                f"study already has {finished_trials} finished trials "
                f"(target: {args.trials})"
            )
    finally:
        write_trials_csv(study, output / "trials.csv")
        complete = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if complete:
            best = study.best_trial
            summary = {
                "study": study.study_name,
                "best_trial": best.number,
                "best_value": best.value,
                "best_params": best.params,
            }
            (output / "best.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
