from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from common import image_paths, read_json, sha256_path, utc_now, write_json_atomic


SCHEMA_VERSION = "robot_tracking_comparison_v1"
METHODS = ("sam3", "robotseg")


def _config(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected {SCHEMA_VERSION}, got {value.get('schema_version')!r}")
    scope = value.get("scope", {})
    forbidden = ("generate_other_entities", "generate_scene_graph", "generate_depth_or_trajectory")
    if any(scope.get(key) is not False for key in forbidden):
        raise ValueError("Robot comparison must remain segmentation-only")
    if scope.get("entity") != "whole_robot" or scope.get("process_every_frame") is not True:
        raise ValueError("Robot comparison must segment the whole robot in every frame")
    return value


def _entries(manifest: Path) -> list[dict[str, Any]]:
    value = read_json(manifest)
    items = value.get("episodes") or value.get("scenes")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty episodes or scenes list")
    return items


def _relative_path(item: dict[str, Any]) -> Path:
    if item.get("relative_path"):
        return Path(item["relative_path"])
    return Path(item["dataset_name"]) / item["scene_id"]


def _fingerprint(frame_paths: list[Path], method: str, config: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode())
    digest.update(method.encode())
    digest.update(json.dumps(config["scope"], sort_keys=True).encode())
    digest.update(json.dumps(config[method], sort_keys=True).encode())
    for frame_path in frame_paths:
        digest.update(frame_path.name.encode())
        digest.update(sha256_path(frame_path).encode())
    return digest.hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _sam3_model(config: dict[str, Any]) -> Any:
    from sam_stage import build_model

    sam_config = {
        "segmenter": {"backend": "sam3"},
        "sam3": {**config["sam3"], "prompt_mode": "text"},
    }
    return build_model(sam_config)


def _sam3_masks(
    model: Any, frames: Path, frame_count: int, config: dict[str, Any]
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    from sam_stage import _run_prompt

    return _run_prompt(model, frames, config["sam3"]["text_prompt"], frame_count)


def _robotseg_model(config: dict[str, Any]) -> Any:
    checkpoint = Path(config["robotseg"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_hash = sha256_path(checkpoint)
    if actual_hash != config["robotseg"]["checkpoint_sha256"]:
        raise ValueError(f"RobotSeg checkpoint hash mismatch: {actual_hash}")
    from robotseg.build_robotseg import build_robotseg_video_predictor

    return build_robotseg_video_predictor(
        config["robotseg"]["model_config"],
        str(checkpoint),
        device="cuda:0",
        mode="eval",
        apply_postprocessing=True,
    )


def _prepare_ordered_frames(frame_paths: list[Path], target: Path) -> None:
    target.mkdir()
    for index, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as image:
            image.convert("RGB").save(
                target / f"{index:05d}.jpg", format="JPEG", quality=100, subsampling=0
            )


def _robotseg_masks(
    model: Any,
    frame_paths: list[Path],
    temporary_frames: Path,
    config: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    _prepare_ordered_frames(frame_paths, temporary_frames)
    category = config["robotseg"]["category"]
    state = model.init_state(
        video_path=str(temporary_frames),
        async_loading_frames=False,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    )
    masks: dict[int, np.ndarray] = {}
    prompt_frame = int(config["scope"]["prompt_frame_index"])
    _, object_ids, initial_logits = model.add_new_robot(
        inference_state=state,
        frame_idx=prompt_frame,
        obj_id=0,
        robot=category,
    )
    initial_ids = list(object_ids)
    if 0 in initial_ids:
        masks[prompt_frame] = _as_numpy(
            initial_logits[initial_ids.index(0)] > 0.0
        ).squeeze().astype(bool)
    for frame_index, output_ids, logits in model.propagate_in_video(
        inference_state=state,
        robot=category,
        start_frame_idx=prompt_frame,
        max_frame_num_to_track=len(frame_paths),
        reverse=False,
    ):
        ids = list(output_ids)
        if 0 in ids:
            masks[int(frame_index)] = _as_numpy(logits[ids.index(0)] > 0.0).squeeze().astype(bool)
    del state
    return masks, {
        "status": "tracked" if masks else "not_found",
        "visible_frame_count": len(masks),
        "category": category,
        "prompt_mode": "official automatic robot prompt generator",
    }


def _guided_refine(mask: np.ndarray, frame_path: Path) -> np.ndarray:
    from utils import guided_refine_mask

    image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read {frame_path}")
    return np.asarray(guided_refine_mask(mask.astype(np.uint8) * 255, image_bgr)) > 0


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _diagnostics(masks: list[np.ndarray]) -> dict[str, Any]:
    if not masks:
        raise ValueError("Diagnostics require at least one mask")
    height, width = masks[0].shape
    diagonal = max(float(np.hypot(width, height)), 1.0)
    frame_area = float(width * height)
    visible = [bool(mask.any()) for mask in masks]
    area_fractions = [float(mask.sum()) / frame_area for mask in masks]
    consecutive_ious: list[float] = []
    centroid_jumps: list[float] = []
    previous_centroid: tuple[float, float] | None = None
    component_counts: list[int] = []
    for index, mask in enumerate(masks):
        if index:
            union = np.logical_or(masks[index - 1], mask).sum()
            if union:
                consecutive_ious.append(float(np.logical_and(masks[index - 1], mask).sum() / union))
        if mask.any():
            ys, xs = np.nonzero(mask)
            centroid = (float(xs.mean()), float(ys.mean()))
            if previous_centroid is not None:
                centroid_jumps.append(
                    float(np.hypot(
                        centroid[0] - previous_centroid[0], centroid[1] - previous_centroid[1]
                    ) / diagonal)
                )
            previous_centroid = centroid
            component_counts.append(max(0, int(cv2.connectedComponents(mask.astype(np.uint8))[0]) - 1))
    return {
        "note": "These are tracking diagnostics without ground-truth masks, not accuracy metrics.",
        "frame_count": len(masks),
        "visible_frame_count": sum(visible),
        "visible_fraction": round(float(np.mean(visible)), 6),
        "median_mask_area_fraction": round(float(np.median(area_fractions)), 6),
        "median_consecutive_mask_iou": round(float(np.median(consecutive_ious)), 6) if consecutive_ious else None,
        "median_normalized_centroid_jump": round(float(np.median(centroid_jumps)), 6) if centroid_jumps else None,
        "median_connected_components": round(float(np.median(component_counts)), 3) if component_counts else None,
    }


def _save_masks_atomic(
    scene_output: Path,
    method: str,
    masks: dict[int, np.ndarray],
    track: dict[str, Any],
    frame_paths: list[Path],
    config: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    masks_root = scene_output / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)
    final_dir = masks_root / method
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{method}.", dir=masks_root))
    normalized: list[np.ndarray] = []
    with Image.open(frame_paths[0]) as first:
        width, height = first.convert("RGB").size
    try:
        for frame_index, frame_path in enumerate(frame_paths):
            mask = masks.get(frame_index)
            if mask is None:
                mask = np.zeros((height, width), dtype=bool)
            else:
                mask = _resize_mask(mask, width, height)
                if method == "robotseg" and config["robotseg"]["guided_refinement"]:
                    mask = _guided_refine(mask, frame_path)
            normalized.append(mask)
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                temporary_dir / f"frame_{frame_index:06d}.png"
            )
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temporary_dir.replace(final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    method_config = config[method]
    report = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "scope": config["scope"],
        "fingerprint": fingerprint,
        "completed_at": utc_now(),
        "source_revision": method_config["source_revision"],
        "checkpoint_sha256": method_config["checkpoint_sha256"],
        "prompt": method_config.get("text_prompt") or method_config.get("category"),
        "guided_refinement": bool(method_config.get("guided_refinement", False)),
        "track": track,
        "diagnostics": _diagnostics(normalized),
    }
    write_json_atomic(scene_output / f"{method}_report.json", report)
    return report


def _method_current(scene_output: Path, method: str, fingerprint: str, frame_count: int) -> bool:
    report_path = scene_output / f"{method}_report.json"
    mask_dir = scene_output / "masks" / method
    if not report_path.is_file() or not mask_dir.is_dir():
        return False
    try:
        report = read_json(report_path)
        mask_count = len(image_paths(mask_dir))
    except Exception:
        return False
    return report.get("fingerprint") == fingerprint and mask_count == frame_count


def segment_manifest(
    backend: str,
    manifest: Path,
    source_root: Path,
    output_root: Path,
    config: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if backend not in METHODS:
        raise ValueError(f"Unknown backend {backend}")
    model = _sam3_model(config) if backend == "sam3" else _robotseg_model(config)
    results = []
    for item in _entries(manifest):
        relative = _relative_path(item)
        frames_dir = source_root / relative / "images"
        scene_output = output_root / relative
        scene_output.mkdir(parents=True, exist_ok=True)
        frame_paths = image_paths(frames_dir)
        fingerprint = _fingerprint(frame_paths, backend, config)
        if not overwrite and _method_current(scene_output, backend, fingerprint, len(frame_paths)):
            results.append({"relative_path": str(relative), "status": "skipped_current"})
            continue
        import torch

        with tempfile.TemporaryDirectory(prefix=".ordered_frames.", dir=scene_output) as temporary:
            staged_frames = Path(temporary) / "frames"
            if backend == "sam3":
                _prepare_ordered_frames(frame_paths, staged_frames)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, track = _sam3_masks(model, staged_frames, len(frame_paths), config)
            else:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, track = _robotseg_masks(model, frame_paths, staged_frames, config)
        _save_masks_atomic(scene_output, backend, masks, track, frame_paths, config, fingerprint)
        results.append({"relative_path": str(relative), "status": "success"})
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": backend,
        "completed_at": utc_now(),
        "scene_count": len(results),
        "success_count": sum(item["status"] in {"success", "skipped_current"} for item in results),
        "results": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / f"batch_{backend}_report.json", summary)
    return summary


def _overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    width: int,
) -> np.ndarray:
    output = frame.copy()
    color_layer = np.zeros_like(output)
    color_layer[:] = color
    output[mask] = cv2.addWeighted(output, 1.0 - alpha, color_layer, alpha, 0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 255, 255), width + 2, cv2.LINE_AA)
    cv2.drawContours(output, contours, -1, color, width, cv2.LINE_AA)
    return output


def _panel(
    frame: np.ndarray,
    mask: np.ndarray | None,
    label: str,
    color: tuple[int, int, int],
    render: dict[str, Any],
) -> np.ndarray:
    panel_width = int(render["panel_width"])
    title_height = int(render["title_height"])
    height, width = frame.shape[:2]
    panel_height = max(1, round(height * panel_width / width))
    resized = cv2.resize(frame, (panel_width, panel_height), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        resized_mask = cv2.resize(
            mask.astype(np.uint8), (panel_width, panel_height), interpolation=cv2.INTER_NEAREST
        ) > 0
        resized = _overlay(
            resized, resized_mask, color, float(render["mask_alpha"]), int(render["boundary_width"])
        )
        status = f"visible | area {100.0 * float(resized_mask.mean()):.1f}%" if resized_mask.any() else "empty mask"
    else:
        status = "unmodified input"
    title = np.full((title_height, panel_width, 3), 24, dtype=np.uint8)
    cv2.putText(title, label, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(title, status, (14, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([title, resized])


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _comparison_frame(
    frame_path: Path,
    sam_mask_path: Path,
    robotseg_mask_path: Path,
    frame_index: int,
    config: dict[str, Any],
) -> np.ndarray:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"OpenCV could not read {frame_path}")
    render = config["render"]
    return np.hstack([
        _panel(frame, None, f"RGB | frame {frame_index:02d}", (230, 230, 230), render),
        _panel(frame, _mask(sam_mask_path), 'SAM3 | text: "robot"', (0, 210, 255), render),
        _panel(frame, _mask(robotseg_mask_path), "RobotSeg | auto: robot", (255, 80, 210), render),
    ])


def _render_scene(
    relative: Path,
    source_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    frame_paths = image_paths(source_root / relative / "images")
    scene_output = output_root / relative
    sam_paths = image_paths(scene_output / "masks" / "sam3")
    robotseg_paths = image_paths(scene_output / "masks" / "robotseg")
    if not (len(frame_paths) == len(sam_paths) == len(robotseg_paths)):
        raise ValueError(f"Frame/mask count mismatch for {relative}")
    with Image.open(frame_paths[0]) as first:
        expected_size = first.convert("RGB").size
    for path in [*frame_paths, *sam_paths, *robotseg_paths]:
        with Image.open(path) as image:
            if image.size != expected_size:
                raise ValueError(f"Dimension mismatch at {path}: {image.size} != {expected_size}")
    first_comparison = _comparison_frame(frame_paths[0], sam_paths[0], robotseg_paths[0], 0, config)
    height, width = first_comparison.shape[:2]
    temporary_video = scene_output / ".visualization.mp4.tmp.mp4"
    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config["render"]["fps"]),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 writer")
    sample_indices = sorted({0, len(frame_paths) // 3, 2 * len(frame_paths) // 3, len(frame_paths) - 1})
    samples: list[np.ndarray] = []
    try:
        for index, (frame_path, sam_path, robotseg_path) in enumerate(
            zip(frame_paths, sam_paths, robotseg_paths)
        ):
            comparison = first_comparison if index == 0 else _comparison_frame(
                frame_path, sam_path, robotseg_path, index, config
            )
            writer.write(comparison)
            if index in sample_indices:
                samples.append(comparison)
    finally:
        writer.release()
    temporary_video.replace(scene_output / "visualization.mp4")
    thumb_width = width // 2
    thumb_height = height // 2
    thumbs = [
        cv2.resize(sample, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        for sample in samples
    ]
    while len(thumbs) < 4:
        thumbs.append(np.zeros((thumb_height, thumb_width, 3), dtype=np.uint8))
    sheet = np.vstack([np.hstack(thumbs[:2]), np.hstack(thumbs[2:4])])
    contact_tmp = scene_output / ".contact_sheet.png.tmp.png"
    if not cv2.imwrite(str(contact_tmp), sheet):
        raise RuntimeError("Could not write contact sheet")
    contact_tmp.replace(scene_output / "contact_sheet.png")
    report = {
        "schema_version": SCHEMA_VERSION,
        "relative_path": str(relative),
        "frame_count": len(frame_paths),
        "frame_dimensions": [expected_size[0], expected_size[1]],
        "methods": {
            method: read_json(scene_output / f"{method}_report.json")["diagnostics"]
            for method in METHODS
        },
        "artifacts": ["visualization.mp4", "contact_sheet.png"],
        "completed_at": utc_now(),
    }
    write_json_atomic(scene_output / "comparison_report.json", report)
    return report


def render_manifest(
    manifest: Path, source_root: Path, output_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    results = []
    for item in _entries(manifest):
        relative = _relative_path(item)
        report = _render_scene(relative, source_root, output_root, config)
        results.append({
            "relative_path": str(relative),
            "status": "success",
            "frame_count": report["frame_count"],
        })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "render_and_validate",
        "completed_at": utc_now(),
        "scene_count": len(results),
        "success_count": sum(item["status"] == "success" for item in results),
        "results": results,
    }
    write_json_atomic(output_root / "batch_render_report.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Whole-robot RobotSeg versus SAM3 text tracking experiment"
    )
    root.add_argument("--config", default="/pipeline/robot_tracking_config.json")
    commands = root.add_subparsers(dest="command", required=True)
    segment = commands.add_parser("segment")
    segment.add_argument("--backend", choices=METHODS, required=True)
    render = commands.add_parser("render")
    for command in (segment, render):
        command.add_argument("--manifest", required=True)
        command.add_argument("--source-root", required=True)
        command.add_argument("--output-root", required=True)
    segment.add_argument("--overwrite", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    config = _config(Path(args.config))
    manifest = Path(args.manifest)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if args.command == "segment":
        result = segment_manifest(
            args.backend, manifest, source_root, output_root, config, args.overwrite
        )
    else:
        result = render_manifest(manifest, source_root, output_root, config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
