from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common import read_json


def _path(value: str | None) -> Path | None:
    return Path(value).resolve() if value else None


def _config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    value = read_json(config_path)
    if value.get("schema_version") == "unified_sgg_config_override_v1":
        base_path = (config_path.parent / value["extends"]).resolve()
        base = read_json(base_path)

        def merge(target: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
            result = dict(target)
            for key, item in override.items():
                if isinstance(item, dict) and isinstance(result.get(key), dict):
                    result[key] = merge(result[key], item)
                else:
                    result[key] = item
            return result

        value = merge(base, value.get("override", {}))
    if value.get("schema_version") != "unified_sgg_config_v1":
        raise ValueError("Unknown configuration schema")
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Goal-guided all-frame video segmentation pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--images")
    source.add_argument("--video")
    goal = prepare.add_mutually_exclusive_group(required=True)
    goal.add_argument("--planning-goal-json")
    goal.add_argument("--planning-goal")
    prepare.add_argument("--overwrite", action="store_true")
    for name in ("entities", "sam", "visualize"):
        item = commands.add_parser(name)
        item.add_argument("--output", required=True)
        item.add_argument("--config", default="/pipeline/config.json")
        item.add_argument("--overwrite", action="store_true")
    batch = commands.add_parser("batch")
    batch.add_argument(
        "--stage", required=True,
        choices=("prepare", "entities", "sam", "visualize"),
    )
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--source-root", required=True)
    batch.add_argument("--output-root", required=True)
    batch.add_argument("--config", default="/pipeline/config.json")
    batch.add_argument("--overwrite", action="store_true")
    batch.add_argument("--fail-fast", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    output = Path(args.output).resolve() if hasattr(args, "output") else None
    if args.command == "prepare":
        from prepare import prepare
        result = prepare(output, _path(args.images), _path(args.video), _path(args.planning_goal_json), args.planning_goal, args.overwrite)
    elif args.command == "entities":
        from qwen import infer_entities
        result = infer_entities(output, _config(args.config), args.overwrite)
    elif args.command == "sam":
        from sam_stage import segment
        result = segment(output, _config(args.config), args.overwrite)
    elif args.command == "visualize":
        from visualize import visualize
        result = visualize(output, _config(args.config), args.overwrite)
    elif args.command == "batch":
        from batch import run_batch
        result = run_batch(
            args.stage, Path(args.manifest), Path(args.source_root), Path(args.output_root),
            _config(args.config), args.overwrite, args.fail_fast,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
