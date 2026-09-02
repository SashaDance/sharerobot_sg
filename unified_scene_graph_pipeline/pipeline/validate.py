from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common import ensure_no_confidence, image_paths, read_json, update_run_report, write_json_atomic


PUBLIC_JSON = ("input.json", "task_spec.json", "tracks.json", "scene_graph.json", "trajectory_3d.json")
FORBIDDEN_KEYS = ("planning_steps", "keyframes", "keyframe", "source_masks", "source_mask", "aliases", "synonyms")


def _forbidden(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden field at {location}.{key}")
            _forbidden(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden(child, f"{location}[{index}]")


def _validate_da3(da3: Path, frame_count: int) -> dict[str, Any]:
    npzs = sorted((da3 / "results_output").glob("frame_*.npz"), key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
    if len(npzs) != frame_count:
        raise ValueError(f"DA3 NPZ count {len(npzs)} != {frame_count}")
    for path in npzs:
        with np.load(path) as data:
            for key in ("depth", "intrinsics"):
                if key not in data or not np.isfinite(data[key]).all():
                    raise ValueError(f"Invalid DA3 {key} in {path.name}")
    pose_lines = [line for line in (da3 / "camera_poses.txt").read_text().splitlines() if line.strip()]
    pose_values = [np.fromstring(line, sep=" ") for line in pose_lines]
    if len(pose_lines) != frame_count or any(values.size != 16 or not np.isfinite(values).all() for values in pose_values):
        raise ValueError("Invalid DA3 camera pose count or shape")
    for path in (da3 / "camera_poses.ply", da3 / "pcd" / "combined_pcd.ply"):
        if not path.is_file() or path.stat().st_size < 200:
            raise ValueError(f"Missing/empty DA3 artifact {path}")
    return {"npz_count": len(npzs), "pose_count": len(pose_lines)}


def validate(output: Path, require_da3: bool = True) -> dict[str, Any]:
    errors = []
    details: dict[str, Any] = {}
    try:
        context = read_json(output / "input.json")
        count, width, height = int(context["frame_count"]), int(context["width"]), int(context["height"])
        frames = image_paths(output / "frames")
        if len(frames) != count or [path.name for path in frames] != [f"frame_{index:06d}.png" for index in range(count)]:
            raise ValueError("Input/output frame correspondence is not exact")
        for path in frames:
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (width, height):
                    raise ValueError(f"Non-normalized frame {path.name}: {image.mode} {image.size}")
        task = read_json(output / "task_spec.json")
        roles = task["roles"]
        if roles.get("robot") is None or roles.get("manipulated_object") is None:
            raise ValueError("Required task role is missing")
        entities = {entity["entity_id"] for entity in task["entities"]}
        tracks = read_json(output / "tracks.json")
        if tracks.get("frame_count") != count or {track["entity_id"] for track in tracks.get("tracks", [])} != entities:
            raise ValueError("Track/entity correspondence mismatch")
        for track in tracks["tracks"]:
            if [row.get("frame_index") for row in track.get("frames", [])] != list(range(count)):
                raise ValueError(f"Track frame correspondence mismatch for {track['entity_id']}")
            if any(row.get("status") not in {"visible", "not_visible", "segmentation_failed"} for row in track["frames"]):
                raise ValueError(f"Missing explicit mask status for {track['entity_id']}")
        for entity_id in entities:
            masks = image_paths(output / "masks" / entity_id)
            if len(masks) != count:
                raise ValueError(f"Mask count mismatch for {entity_id}")
            for path in masks:
                with Image.open(path) as image:
                    if image.size != (width, height) or image.mode != "L":
                        raise ValueError(f"Mask geometry mismatch: {path}")
        graph = read_json(output / "scene_graph.json")
        if graph["frame_count"] != count or [row["frame_index"] for row in graph["frames"]] != list(range(count)):
            raise ValueError("Expected exactly one ordered graph per input frame")
        for row in graph["frames"]:
            for edge in row["state_edges"]:
                if edge["subject"] not in entities or (
                    edge["object"] not in entities and not (edge["relation"] in {"open", "closed"} and edge["object"] is None)
                ):
                    raise ValueError(f"Invalid graph entity reference: {edge}")
            for action in row["actions"]:
                if action["actor"] not in entities or (action["object"] is not None and action["object"] not in entities):
                    raise ValueError(f"Invalid action entity reference: {action}")
        trajectory = read_json(output / "trajectory_3d.json")
        if [point["frame_index"] for point in trajectory["points"]] != list(range(count)):
            raise ValueError("Trajectory/frame correspondence mismatch")
        for name in PUBLIC_JSON:
            document = read_json(output / name)
            ensure_no_confidence(document, name)
            _forbidden(document, name)
        if require_da3:
            details["da3"] = _validate_da3(output / "da3", count)
        for artifact in (output / "visualization.mp4", output / "contact_sheet.jpg", output / "run_report.json"):
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise ValueError(f"Missing final scene artifact: {artifact.name}")
        with Image.open(output / "contact_sheet.jpg") as contact_sheet:
            contact_sheet.verify()
        details.update({"frame_count": count, "entity_count": len(entities), "track_count": len(tracks["tracks"])})
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    result = {"schema_version": "unified_sgg_validation_v1", "status": "passed" if not errors else "failed", "errors": errors, "details": details}
    write_json_atomic(output / "validation_report.json", result)
    update_run_report(output, "validate", result["status"], {"errors": errors})
    if errors:
        raise RuntimeError(errors[0])
    return result
