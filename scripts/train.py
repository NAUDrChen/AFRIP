from __future__ import annotations

import argparse
import json

from afrip.utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="AFRIP training entry stub")
    parser.add_argument("--config", required=True, help="Experiment config path")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = {
        "experiment": config.get("experiment", {}).get("name", "unknown"),
        "task": config.get("dataset", {}).get("task", "unknown"),
        "detector": config.get("detector", {}).get("type", "unknown"),
        "tracker": config.get("tracker", {}).get("type", "unknown"),
        "trainer": config.get("strategy", {}).get("trainer", {}).get("type", "unknown"),
        "work_dir": config.get("runtime", {}).get("work_dir", "outputs/default"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
