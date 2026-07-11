#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freeciv_sim.rules.ruleset_loader import export_ruleset_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the active Freeciv ruleset for inspection.")
    parser.add_argument("--output-dir", default="config", type=Path)
    args = parser.parse_args()
    export_ruleset_config(args.output_dir)


if __name__ == "__main__":
    main()
