#!/usr/bin/env python3
"""Convert Freeciv ACTLOG lines from server.log into JSONL files.

Usage examples:
  scripts/export_actionlog_jsonl.py --run-dir runs/20260306-124108
  scripts/export_actionlog_jsonl.py --log-file runs/20260306-124108/logs/server.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator

KV_PATTERN = re.compile(r'([A-Za-z0-9_]+)=("([^"\\]|\\.)*"|[^\t ]+)')
INT_PATTERN = re.compile(r"^-?\d+$")
FLOAT_PATTERN = re.compile(r"^-?\d+\.\d+$")


def parse_value(raw: str):
    if raw in {"null", "nil"}:
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    if INT_PATTERN.fullmatch(raw):
        return int(raw)
    if FLOAT_PATTERN.fullmatch(raw):
        return float(raw)
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return json.loads(raw)
    return raw


def iter_actlog_records(log_file: Path, run_id: str | None) -> Iterator[dict]:
    with log_file.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            marker = line.find("ACTLOG")
            if marker < 0:
                continue

            payload = line[marker + len("ACTLOG") :].strip()
            record: dict = {}

            for match in KV_PATTERN.finditer(payload):
                key = match.group(1)
                raw_value = match.group(2)
                record[key] = parse_value(raw_value)

            if not record:
                continue

            if run_id is not None:
                record["run_id"] = run_id
            record["_source_line"] = lineno
            yield record


def resolve_paths(run_dir: Path | None, log_file: Path | None):
    if run_dir is not None:
        run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run dir not found: {run_dir}")
        log_file = run_dir / "logs" / "server.log"
        if not log_file.is_file():
            raise FileNotFoundError(f"server log not found: {log_file}")
        out_jsonl = run_dir / "logs" / "actionlog.jsonl"
        out_actions = run_dir / "logs" / "actionlog_actions.jsonl"
        run_id = run_dir.name
        return log_file, out_jsonl, out_actions, run_id

    assert log_file is not None
    log_file = log_file.resolve()
    if not log_file.is_file():
        raise FileNotFoundError(f"log file not found: {log_file}")
    out_jsonl = log_file.with_suffix(".jsonl")
    out_actions = log_file.with_name(f"{log_file.stem}_actions.jsonl")
    return log_file, out_jsonl, out_actions, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Freeciv ACTLOG lines to JSONL."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path, help="Run directory (contains logs/server.log).")
    group.add_argument("--log-file", type=Path, help="Path to server.log.")
    parser.add_argument(
        "--no-source-line",
        action="store_true",
        help="Do not include '_source_line' in output records.",
    )
    parser.add_argument("--out-jsonl", type=Path, help="Path for all ACTLOG records.")
    parser.add_argument("--out-actions", type=Path, help="Path for action_finished_* records.")
    args = parser.parse_args()

    log_file, out_jsonl, out_actions, run_id = resolve_paths(args.run_dir, args.log_file)
    if args.out_jsonl is not None:
        out_jsonl = args.out_jsonl
    if args.out_actions is not None:
        out_actions = args.out_actions
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_actions.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    actions = 0
    with out_jsonl.open("w", encoding="utf-8") as all_fh, out_actions.open(
        "w", encoding="utf-8"
    ) as action_fh:
        for record in iter_actlog_records(log_file, run_id):
            if args.no_source_line:
                record.pop("_source_line", None)

            all_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

            if str(record.get("event", "")).startswith("action_finished_"):
                action_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                actions += 1

    print(f"log_file: {log_file}")
    print(f"actionlog_jsonl: {out_jsonl} ({total} lines)")
    print(f"actions_jsonl: {out_actions} ({actions} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
