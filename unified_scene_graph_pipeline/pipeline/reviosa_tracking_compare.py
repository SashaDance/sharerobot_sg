from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer


SCHEMA_VERSION = "reviosa_interaction_tracking_v1"
FRAME_RE = re.compile(r"(\d+)(?=\.[^.]+$)")
ACTOR_COLOR = (230, 57, 70)
TARGET_COLOR = (69, 123, 157)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _numeric_key(path: Path) -> tuple[int, str]:
    match = FRAME_RE.search(path.name)
    return (int(match.group(1)) if match else 10**12, path.name)


def _frames(directory: Path) -> list[Path]:
    result = sorted(
        [*directory.glob("*.png"), *directory.glob("*.jpg"), *directory.glob("*.jpeg")],
        key=_numeric_key,
    )
    if not result:
        raise ValueError(f"No frames found under {directory}")
    return result


def _entries(manifest: Path) -> list[dict[str, Any]]:
    value = _read_json(manifest)
    result = value.get("episodes") or value.get("scenes")
    if not isinstance(result, list) or not result:
        raise ValueError("Manifest must contain a non-empty episodes or scenes list")
    return result


def _relative_path(item: dict[str, Any]) -> Path:
    if item.get("relative_path"):
        return Path(item["relative_path"])
    return Path(item["dataset_name"]) / item["scene_id"]


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected {SCHEMA_VERSION}, got {config.get('schema_version')!r}")
    template = config.get("prompt", {}).get("template", "")
    if template.count("{planning_goal}") != 1:
        raise ValueError("Prompt template must contain exactly one {planning_goal}")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_hash(model_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in model_root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(model_root)).encode())
        digest.update(str(path.stat().st_size).encode())
        if path.suffix in {".json", ".py"}:
            digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _model(config: dict[str, Any]) -> tuple[Any, Any, str]:
    model_config = config["model"]
    model_path = Path(model_config["path"])
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=bool(model_config["use_flash_attention"]),
        trust_remote_code=True,
        local_files_only=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=False, local_files_only=True
    )
    return model, tokenizer, _snapshot_hash(model_path)


def _normalize_masks(value: Any, frame_count: int, height: int, width: int) -> list[np.ndarray]:
    if not value:
        return [np.zeros((height, width), dtype=bool) for _ in range(frame_count)]
    masks = np.asarray(value[0], dtype=bool)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.shape[0] != frame_count:
        raise ValueError(f"Expected {frame_count} masks, got {masks.shape[0]}")
    result = []
    for mask in masks:
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        result.append(mask.astype(bool))
    return result


def _diagnostics(masks: list[np.ndarray]) -> dict[str, Any]:
    frame_area = float(masks[0].shape[0] * masks[0].shape[1])
    areas = [float(mask.sum()) / frame_area for mask in masks]
    visible = [mask.any() for mask in masks]
    ious = []
    for previous, current in zip(masks, masks[1:]):
        union = np.logical_or(previous, current).sum()
        if union:
            ious.append(float(np.logical_and(previous, current).sum() / union))
    return {
        "note": "Diagnostics measure temporal behavior, not segmentation accuracy.",
        "frame_count": len(masks),
        "visible_frame_count": int(sum(visible)),
        "median_mask_area_fraction": round(float(np.median(areas)), 6),
        "median_consecutive_iou": round(float(np.median(ious)), 6) if ious else None,
    }


def _save_masks(directory: Path, masks: list[np.ndarray]) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        for index, mask in enumerate(masks):
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                temporary / f"frame_{index:06d}.png"
            )
        if directory.exists():
            shutil.rmtree(directory)
        temporary.replace(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _boundary(mask: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones((max(1, width), max(1, width)), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
    edge = mask & ~eroded
    return cv2.dilate(edge.astype(np.uint8), kernel) > 0


def _panel(frame: np.ndarray, mask: np.ndarray | None, title: str, color: tuple[int, int, int], config: dict[str, Any]) -> np.ndarray:
    panel_width = int(config["panel_width"])
    title_height = int(config["title_height"])
    height = max(1, round(frame.shape[0] * panel_width / frame.shape[1]))
    panel = cv2.resize(frame, (panel_width, height), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        resized_mask = cv2.resize(mask.astype(np.uint8), (panel_width, height), interpolation=cv2.INTER_NEAREST) > 0
        layer = np.zeros_like(panel)
        layer[:] = color[::-1]
        alpha = float(config["mask_alpha"])
        panel[resized_mask] = cv2.addWeighted(panel, 1.0 - alpha, layer, alpha, 0)[resized_mask]
        panel[_boundary(resized_mask, int(config["boundary_width"]))] = color[::-1]
    canvas = np.full((height + title_height, panel_width, 3), 18, dtype=np.uint8)
    canvas[title_height:] = panel
    cv2.putText(canvas, title, (9, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.64, color[::-1], 2, cv2.LINE_AA)
    return canvas


def _render_video(
    destination: Path,
    frames: list[Path],
    actor_masks: list[np.ndarray],
    target_masks: list[np.ndarray],
    goal: str,
    prompt: str,
    config: dict[str, Any],
) -> None:
    render = config["render"]
    with tempfile.TemporaryDirectory(prefix="reviosa_render_") as temporary_name:
        temporary = Path(temporary_name)
        for index, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not read {frame_path}")
            panels = [
                _panel(frame, None, f"Raw | frame {index}", (230, 230, 230), render),
                _panel(frame, actor_masks[index], "ReVIOSa actor -> robot", ACTOR_COLOR, render),
                _panel(frame, target_masks[index], "ReVIOSa target -> manipulated object", TARGET_COLOR, render),
            ]
            row = np.concatenate(panels, axis=1)
            header = np.full((74, row.shape[1], 3), 18, dtype=np.uint8)
            cv2.putText(header, f"Goal: {goal}"[:190], (9, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(header, f"Global prompt: {prompt.replace(chr(10), ' ')}"[:190], (9, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (205, 205, 205), 1, cv2.LINE_AA)
            cv2.imwrite(str(temporary / f"frame_{index:06d}.jpg"), np.concatenate([header, row], axis=0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_video = destination.with_name(f".{destination.name}.tmp.mp4")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(render["fps"]), "-i", str(temporary / "frame_%06d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(temporary_video),
            ],
            check=True,
        )
        temporary_video.replace(destination)


def _run_scene(
    model: Any,
    tokenizer: Any,
    item: dict[str, Any],
    source_root: Path,
    output_root: Path,
    config: dict[str, Any],
    snapshot_hash: str,
) -> dict[str, Any]:
    relative = _relative_path(item)
    source = source_root / relative
    destination = output_root / relative
    destination.mkdir(parents=True, exist_ok=True)
    frame_paths = _frames(source / "images")
    goal_document = _read_json(source / "planning_goal.json")
    goal = str(goal_document["planning_goal"]).strip()
    prompt = config["prompt"]["template"].format(planning_goal=goal)
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("All video frames must have stable dimensions")
    with torch.inference_mode():
        result = model.predict_forward(video=images, text=prompt, tokenizer=tokenizer)
    actor_masks = _normalize_masks(result.get("prediction_masks_act"), len(images), height, width)
    target_masks = _normalize_masks(result.get("prediction_masks_tar"), len(images), height, width)
    masks_root = destination / "masks"
    masks_root.mkdir(exist_ok=True)
    _save_masks(masks_root / "actor", actor_masks)
    _save_masks(masks_root / "target", target_masks)
    report = {
        "schema_version": SCHEMA_VERSION,
        "completed_at": datetime.now(UTC).isoformat(),
        "relative_path": str(relative),
        "planning_goal": goal,
        "prompt": prompt,
        "model": config["model"],
        "snapshot_manifest_hash": snapshot_hash,
        "prediction_text": result.get("prediction"),
        "actor": {"mapped_role": config["prompt"]["actor_role"], **_diagnostics(actor_masks)},
        "target": {"mapped_role": config["prompt"]["target_role"], **_diagnostics(target_masks)},
    }
    _write_json_atomic(destination / "reviosa_report.json", report)
    _render_video(destination / "comparison.mp4", frame_paths, actor_masks, target_masks, goal, prompt, config)
    for image in images:
        image.close()
    return {"relative_path": str(relative), "status": "success"}


def run(manifest: Path, source_root: Path, output_root: Path, config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    model, tokenizer, snapshot_hash = _model(config)
    results = []
    for item in _entries(manifest):
        try:
            results.append(_run_scene(model, tokenizer, item, source_root, output_root, config, snapshot_hash))
        except Exception as error:
            results.append({
                "relative_path": str(_relative_path(item)),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            })
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "completed_at": datetime.now(UTC).isoformat(),
        "scene_count": len(results),
        "success_count": sum(item["status"] == "success" for item in results),
        "failure_count": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    _write_json_atomic(output_root / "batch_report.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ReVIOSa interaction masks on a fixed manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("/pipeline/reviosa_tracking_config.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.source_root, args.output_root, args.config), indent=2))


if __name__ == "__main__":
    main()
