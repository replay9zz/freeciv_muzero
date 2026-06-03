#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from collections import Counter
from pathlib import Path


STEP_RE = re.compile(
    r"\[turn (?P<turn>\d+) step (?P<step>\d+)\] "
    r"action=(?P<action>\S+) "
    r"enemy_units=(?P<enemy_units>\d+) "
    r"enemy_cities=(?P<enemy_cities>\d+) "
    r"civ_score=(?P<civ_score>-?\d+(?:\.\d+)?)"
    r"(?: fc_score=(?P<fc_score>-?\d+(?:\.\d+)?))?"
    r"(?: fc_win=(?P<fc_win>\S+))?"
)
ISSUE_RE = re.compile(r"\b(Traceback|Exception|Error|Warning)\b")


def _read_steps(path: Path) -> list[dict]:
    steps = []
    if not path.exists():
        return steps
    for line in path.read_text(errors="replace").splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        item = match.groupdict()
        item["turn"] = int(item["turn"])
        item["step"] = int(item["step"])
        item["enemy_units"] = int(item["enemy_units"])
        item["enemy_cities"] = int(item["enemy_cities"])
        item["civ_score"] = float(item["civ_score"])
        item["fc_score"] = (
            float(item["fc_score"]) if item.get("fc_score") is not None else None
        )
        steps.append(item)
    return steps


def _count_issues(*paths: Path) -> int:
    count = 0
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if ISSUE_RE.search(line):
                count += 1
    return count


def _summarize_game(game_dir: Path) -> dict:
    steps = _read_steps(game_dir / "eval.log")
    last = steps[-1] if steps else {}
    action_counts = Counter(step["action"] for step in steps)
    heatmap_videos = list((game_dir / "heatmaps" / "videos").glob("*.mp4"))
    return {
        "game": game_dir.name,
        "turn": last.get("turn"),
        "step": last.get("step"),
        "civ_score": last.get("civ_score"),
        "fc_score": last.get("fc_score"),
        "fc_win": last.get("fc_win"),
        "final_enemy_units": last.get("enemy_units"),
        "final_enemy_cities": last.get("enemy_cities"),
        "seen_enemy_units": int(any(step["enemy_units"] > 0 for step in steps)),
        "seen_enemy_cities": int(any(step["enemy_cities"] > 0 for step in steps)),
        "build_city": action_counts.get("build_city_u0", 0),
        "passes": sum(count for action, count in action_counts.items() if action == "pass"),
        "moves": sum(count for action, count in action_counts.items() if action.startswith("move_")),
        "productions": sum(
            count for action, count in action_counts.items() if action.startswith("produce_")
        ),
        "agent_video": int((game_dir / "eval-agent.mp4").exists()),
        "global_video": int((game_dir / "eval-global.mp4").exists()),
        "heatmap_videos": len(heatmap_videos),
        "issues": _count_issues(game_dir / "eval.log", game_dir / "runner.log"),
    }


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    game_dirs = sorted(path for path in args.run_dir.glob("game-*") if path.is_dir())
    rows = [_summarize_game(path) for path in game_dirs]
    scored = [row["civ_score"] for row in rows if row["civ_score"] is not None]
    turns = [row["turn"] for row in rows if row["turn"] is not None]

    print(f"run: {args.run_dir}")
    print(f"games: {len(rows)}")
    if scored:
        print(
            "civ_score: "
            f"mean={statistics.fmean(scored):.2f} "
            f"median={statistics.median(scored):.2f} "
            f"min={min(scored):.2f} "
            f"max={max(scored):.2f}"
        )
    if turns:
        completed = sum(1 for turn in turns if turn >= max(turns))
        print(f"turns: max={max(turns)} completed_max_turn={completed}/{len(rows)}")
    print(f"final_enemy_city_games: {sum(1 for row in rows if row['final_enemy_cities'])}/{len(rows)}")
    print(f"seen_enemy_city_games: {sum(row['seen_enemy_cities'] for row in rows)}/{len(rows)}")
    print(f"seen_enemy_unit_games: {sum(row['seen_enemy_units'] for row in rows)}/{len(rows)}")
    print(f"agent_videos: {sum(row['agent_video'] for row in rows)}/{len(rows)}")
    print(f"global_videos: {sum(row['global_video'] for row in rows)}/{len(rows)}")
    print(f"heatmap_videos: {sum(row['heatmap_videos'] for row in rows)}")
    print(f"issues: {sum(row['issues'] for row in rows)}")
    print()
    print("game,turn,step,civ_score,final_enemy_units,final_enemy_cities,seen_enemy_cities,issues")
    for row in rows:
        print(
            ",".join(
                [
                    str(row["game"]),
                    str(row["turn"]),
                    str(row["step"]),
                    _fmt_float(row["civ_score"]),
                    str(row["final_enemy_units"]),
                    str(row["final_enemy_cities"]),
                    str(row["seen_enemy_cities"]),
                    str(row["issues"]),
                ]
            )
        )

    if args.csv_out:
        _write_csv(rows, args.csv_out)
        print()
        print(f"csv: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
