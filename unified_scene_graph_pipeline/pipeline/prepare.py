from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from common import image_paths, normalized_frame_name, read_json, update_run_report, write_json_atomic


def planning_goal(source: Path | None, explicit: str | None) -> tuple[str, dict[str, Any]]:
    if explicit and explicit.strip():
        return explicit.strip(), {"type": "explicit_text"}
    if source is None:
        raise ValueError("A planning goal or planning-goal JSON is required")
    value = read_json(source)
    goal = value.get("planning_goal") if isinstance(value, dict) else None
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError(f"No planning_goal in {source}")
    allowed = {
        "schema_version", "episode_id", "dataset_name", "scene_id", "planning_goal",
        "frame_count", "source_dataset", "source_subset",
    }
    forbidden = sorted(set(value) - allowed)
    if forbidden:
        raise ValueError(f"Planning-goal input contains forbidden extra fields: {forbidden}")
    return goal.strip(), {
        "type": "planning_goal_json",
        "episode_id": value.get("episode_id"),
        "dataset_name": value.get("dataset_name"),
        "scene_id": value.get("scene_id"),
    }


def decode_video(video: Path, temporary: Path) -> list[Path]:
    if not video.is_file():
        raise FileNotFoundError(video)
    pattern = temporary / "frame_%09d.png"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vsync", "0", str(pattern),
    ]
    subprocess.run(command, check=True)
    return image_paths(temporary)


def prepare(
    output: Path,
    images: Path | None,
    video: Path | None,
    goal_json: Path | None,
    goal_text: str | None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if (images is None) == (video is None):
        raise ValueError("Specify exactly one of --images or --video")
    context_path = output / "input.json"
    if context_path.is_file() and not overwrite:
        existing = read_json(context_path)
        try:
            existing_frames = image_paths(output / "frames")
            if len(existing_frames) == int(existing["frame_count"]):
                return existing
        except Exception:
            pass
    if output.exists() and not overwrite and not context_path.is_file():
        visible = {path.name for path in output.iterdir() if not path.name.startswith(".")}
        if visible - {"frames"}:
            raise FileExistsError(f"Refusing non-empty unprepared output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frames_out = Path(tempfile.mkdtemp(prefix=".frames.", dir=output))
    frames_out.chmod(0o755)
    goal, source_metadata = planning_goal(goal_json, goal_text)
    with tempfile.TemporaryDirectory(prefix="unified_sgg_decode_") as temporary_name:
        temporary = Path(temporary_name)
        sources = image_paths(images) if images is not None else decode_video(video, temporary)
        dimensions = set()
        for index, source in enumerate(sources):
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                dimensions.add(rgb.size)
                rgb.save(frames_out / normalized_frame_name(index), format="PNG")
    if len(dimensions) != 1:
        raise ValueError(f"Frame dimensions vary within video: {sorted(dimensions)}")
    width, height = next(iter(dimensions))
    context = {
        "schema_version": "unified_sgg_input_v1",
        "planning_goal": goal,
        "frame_count": len(sources),
        "width": width,
        "height": height,
        "frame_pattern": "frames/frame_%06d.png",
        "source": {
            **source_metadata,
            "input_kind": "image_directory" if images is not None else "video",
            "input_path": str((images or video).resolve()),
        },
    }
    final_frames = output / "frames"
    backup = None
    if final_frames.exists():
        backup = output / ".frames.backup"
        if backup.exists():
            shutil.rmtree(backup)
        final_frames.replace(backup)
    frames_out.replace(final_frames)
    if backup is not None:
        shutil.rmtree(backup)
    write_json_atomic(context_path, context)
    update_run_report(output, "prepare", "success", {"frame_count": len(sources), "input_kind": context["source"]["input_kind"]})
    return context
