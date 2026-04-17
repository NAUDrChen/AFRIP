from __future__ import annotations

import argparse
import json
from pathlib import Path

from afrip.utils import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AFRIP utility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show-config", help="Load and print a merged experiment config")
    show_parser.add_argument("--config", required=True, help="Path to the experiment config file")

    tree_parser = subparsers.add_parser("show-layout", help="Print the recommended AFRIP project layout")
    tree_parser.add_argument("--root", default=".", help="Project root path")

    return parser


def show_layout(root: str) -> int:
    root_path = Path(root).resolve()
    for item in [
        "configs/",
        "configs/base/",
        "configs/datasets/",
        "configs/detectors/",
        "configs/trackers/",
        "configs/strategies/",
        "configs/experiments/",
        "docs/",
        "outputs/",
        "scripts/",
        "src/afrip/",
        "tests/",
    ]:
        print(root_path / item)
    return 0


def show_config(config: str) -> int:
    merged = load_config(config)
    print(json.dumps(merged, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "show-config":
        return show_config(args.config)
    if args.command == "show-layout":
        return show_layout(args.root)

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
