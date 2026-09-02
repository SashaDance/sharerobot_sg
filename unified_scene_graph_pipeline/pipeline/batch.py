from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from common import read_json, update_run_report, utc_now, write_json_atomic


def entries(manifest: Path) -> list[dict[str, Any]]:
    value = read_json(manifest)
    items = value.get("episodes") or value.get("scenes")
    if not isinstance(items, list):
        raise ValueError("Manifest must contain episodes or scenes list")
    return items


def paths(item: dict[str, Any], source_root: Path, output_root: Path) -> tuple[Path, Path]:
    relative = item.get("relative_path")
    if not relative:
        dataset = item.get("dataset_name")
        scene = item.get("scene_id")
        if not dataset or not scene:
            raise ValueError(f"Manifest item lacks relative_path: {item}")
        relative = f"{dataset}/{scene}"
    return source_root / relative, output_root / relative


def run_batch(
    stage: str,
    manifest: Path,
    source_root: Path,
    output_root: Path,
    config: dict[str, Any],
    overwrite: bool = False,
    fail_fast: bool = False,
    require_da3: bool = True,
) -> dict[str, Any]:
    results = []
    shared_sam_model = None
    if stage == "sam":
        from sam_stage import build_model
        shared_sam_model = build_model(config)
    for item in entries(manifest):
        source, output = paths(item, source_root, output_root)
        try:
            if stage == "prepare":
                from prepare import prepare
                result = prepare(output, source / "images", None, source / "planning_goal.json", None, overwrite)
            elif stage == "entities":
                from qwen import infer_entities
                result = infer_entities(output, config, overwrite)
            elif stage == "sam":
                from sam_stage import segment
                result = segment(output, config, overwrite, shared_sam_model)
            elif stage == "graph":
                from qwen import infer_graph
                result = infer_graph(output, config, overwrite)
            elif stage == "trajectory":
                from trajectory import compute_trajectory
                result = compute_trajectory(output, config, overwrite)
            elif stage == "render":
                from render import render
                result = render(output, config, overwrite)
            elif stage == "validate":
                from validate import validate
                result = validate(output, require_da3)
            else:
                raise ValueError(f"Unsupported batch stage: {stage}")
            results.append({"relative_path": str(output.relative_to(output_root)), "status": "success"})
        except Exception as error:
            results.append({"relative_path": str(output.relative_to(output_root)), "status": "failed", "error": f"{type(error).__name__}: {error}"})
            output.mkdir(parents=True, exist_ok=True)
            update_run_report(output, stage, "failed", {"error": f"{type(error).__name__}: {error}"})
            if fail_fast:
                raise
    summary = {
        "schema_version": "unified_sgg_batch_report_v1",
        "stage": stage,
        "completed_at": utc_now(),
        "scene_count": len(results),
        "success_count": sum(item["status"] == "success" for item in results),
        "failure_count": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / f"batch_{stage}_report.json", summary)
    return summary
