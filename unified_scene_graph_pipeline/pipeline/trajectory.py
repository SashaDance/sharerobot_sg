from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common import (
    complete_stage,
    config_hash,
    read_json,
    stage_current,
    stage_fingerprint,
    update_run_report,
    write_json_atomic,
)


def _load_poses(path: Path) -> list[np.ndarray]:
    poses = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = np.fromstring(line, sep=" ", dtype=np.float64)
        if values.size != 16:
            raise ValueError(f"{path}:{line_number}: expected 16 pose values, got {values.size}")
        poses.append(values.reshape(4, 4))
    return poses


def _point(mask_path: Path, npz_path: Path, camera_to_world: np.ndarray) -> dict[str, Any]:
    with np.load(npz_path) as data:
        depth = np.asarray(data["depth"], dtype=np.float64).squeeze()
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64).reshape(3, 3)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if mask.shape != depth.shape:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST
            )
        ) > 0
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not mask.any():
        return {"status": "null", "reason": "mask_unavailable", "camera_xyz": None, "world_xyz": None}
    if not valid.any():
        return {"status": "null", "reason": "depth_unavailable", "camera_xyz": None, "world_xyz": None}
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs]
    camera_points = np.column_stack(((xs - intrinsics[0, 2]) * z / intrinsics[0, 0], (ys - intrinsics[1, 2]) * z / intrinsics[1, 1], z))
    camera_xyz = np.median(camera_points, axis=0)
    world_xyz = camera_to_world @ np.r_[camera_xyz, 1.0]
    return {
        "status": "valid",
        "reason": None,
        "camera_xyz": [round(float(value), 7) for value in camera_xyz],
        "world_xyz": [round(float(value), 7) for value in world_xyz[:3]],
        "masked_valid_pixel_count": int(valid.sum()),
        "median_depth": round(float(np.median(z)), 7),
    }


def compute_trajectory(output: Path, config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    dependencies = [output / "input.json", output / "task_spec.json", output / "tracks.json", output / "da3"]
    fingerprint = stage_fingerprint(config, dependencies, "trajectory")
    target = output / "trajectory_3d.json"
    if not overwrite and stage_current(output, "trajectory", fingerprint, [target]):
        return read_json(target)
    context = read_json(output / "input.json")
    task = read_json(output / "task_spec.json")
    entity_id = task["roles"]["manipulated_object"]
    if entity_id is None:
        raise ValueError("Required manipulated_object role is null")
    poses = _load_poses(output / "da3" / "camera_poses.txt")
    if len(poses) != int(context["frame_count"]):
        raise ValueError(f"Expected {context['frame_count']} DA3 camera poses, got {len(poses)}")
    points = []
    for index, camera_to_world in enumerate(poses):
        try:
            item = _point(
                output / "masks" / entity_id / f"frame_{index:06d}.png",
                output / "da3" / "results_output" / f"frame_{index}.npz",
                camera_to_world,
            )
        except FileNotFoundError as error:
            item = {"status": "null", "reason": f"artifact_missing:{error.filename}", "camera_xyz": None, "world_xyz": None}
        points.append({"frame_index": index, **item})
    result = {
        "schema_version": "unified_sgg_trajectory_3d_v1",
        "entity_id": entity_id,
        "coordinate_system": "DA3 world coordinates from camera-to-world poses",
        "estimator": "component-wise median of all valid masked back-projected depth pixels",
        "points": points,
        "provenance": {
            "model": config["da3"]["model_name"],
            "source_revision": config["da3"]["source_revision"],
            "checkpoint_sha256": config["da3"]["checkpoint_sha256"],
            "config_hash": config_hash(config),
        },
    }
    write_json_atomic(target, result)
    complete_stage(output, "trajectory", fingerprint, {"valid_points": sum(point["status"] == "valid" for point in points)})
    update_run_report(output, "trajectory", "success", {"valid_points": sum(point["status"] == "valid" for point in points)})
    return result
