from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import complete_stage, image_paths, read_json, stage_current, stage_fingerprint, update_run_report


ROLE_COLORS = {
    "robot": (230, 57, 70),
    "manipulated_object": (69, 123, 157),
    "initial_support": (42, 157, 143),
    "target": (244, 162, 97),
    "whole_parent": (138, 43, 226),
}
ROLE_MARKERS = {
    "robot": "R",
    "manipulated_object": "M",
    "initial_support": "I",
    "target": "T",
    "whole_parent": "P",
}


def _mask_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    """Return a deterministic boundary band without adding another dependency."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = mask & ~eroded
    for _ in range(max(0, width - 1)):
        padded_boundary = np.pad(boundary, 1, mode="constant", constant_values=False)
        boundary = (
            padded_boundary[1:-1, 1:-1]
            | padded_boundary[:-2, 1:-1]
            | padded_boundary[2:, 1:-1]
            | padded_boundary[1:-1, :-2]
            | padded_boundary[1:-1, 2:]
        )
    return boundary


def _fit_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    while text and draw.textlength(text + suffix, font=font) > max_width:
        text = text[:-1]
    return text + suffix


def _draw_frame(
    output: Path,
    index: int,
    task: dict[str, Any],
    graph: dict[str, Any],
    render_config: dict[str, Any] | None = None,
) -> Image.Image:
    render_config = render_config or {}
    mask_alpha = float(render_config.get("mask_alpha", 0.48))
    boundary_width = int(render_config.get("boundary_width", 2))
    min_width = int(render_config.get("min_width", 640))
    image = Image.open(output / "frames" / f"frame_{index:06d}.png").convert("RGB")
    pixels = np.asarray(image).copy()
    labels = []
    visible_entities: set[str] = set()
    for entity in task["entities"]:
        mask = np.asarray(Image.open(output / "masks" / entity["entity_id"] / f"frame_{index:06d}.png").convert("L")) > 0
        if not mask.any():
            continue
        visible_entities.add(entity["entity_id"])
        color_tuple = ROLE_COLORS.get(entity["role"], (210, 210, 210))
        color = np.asarray(color_tuple)
        pixels[mask] = ((1.0 - mask_alpha) * pixels[mask] + mask_alpha * color).astype(np.uint8)
        pixels[_mask_boundary(mask, boundary_width)] = color
        ys, xs = np.nonzero(mask)
        labels.append((
            int(xs.min()), int(ys.min()), ROLE_MARKERS.get(entity["role"], "?"), color_tuple,
        ))
    rendered = Image.fromarray(pixels)
    scale = max(1.0, min_width / rendered.width)
    if scale > 1:
        rendered = rendered.resize(
            (round(rendered.width * scale), round(rendered.height * scale)),
            Image.Resampling.LANCZOS,
        )
        labels = [(round(x * scale), round(y * scale), label, color) for x, y, label, color in labels]

    font = ImageFont.load_default()
    line_height = 15
    header_lines = [f"Goal: {task.get('planning_goal', '')}"]
    for entity in task["entities"]:
        marker = ROLE_MARKERS.get(entity["role"], "?")
        status = "visible" if entity["entity_id"] in visible_entities else "not visible"
        header_lines.append(
            f"{marker} {entity['role']} | canonical: {entity['canonical_name']} | "
            f"SAM3 prompt: \"{entity['sam_prompt']}\" | {status}"
        )
    header_height = 10 + line_height * len(header_lines)

    row = graph["frames"][index]
    edge_lines = [f"{edge['subject']} {edge['relation']} {edge['object'] or ''}".rstrip() for edge in row["state_edges"]]
    action_lines = [f"{action['actor']} {action['action']} {action['object'] or ''}".rstrip() for action in row["actions"]]
    graph_lines = [f"frame {index}", *(f"S: {line}" for line in edge_lines), *(f"A: {line}" for line in action_lines)]
    graph_height = 8 + line_height * max(1, len(graph_lines))

    canvas_width = rendered.width + (rendered.width % 2)
    canvas_height = header_height + rendered.height + graph_height
    canvas_height += canvas_height % 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (16, 18, 22))
    canvas.paste(rendered, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.line((0, header_height - 1, canvas_width, header_height - 1), fill=(75, 80, 90), width=1)
    for line_index, line in enumerate(header_lines):
        y = 5 + line_index * line_height
        if line_index == 0:
            color = (245, 245, 245)
        else:
            entity = task["entities"][line_index - 1]
            color = ROLE_COLORS.get(entity["role"], (210, 210, 210))
        draw.text((6, y), _fit_text(line, draw, font, canvas_width - 12), fill=color, font=font)
    for x, y, label, color in labels:
        x = min(x, canvas_width - 16)
        y = min(y + header_height, header_height + rendered.height - line_height)
        box = draw.textbbox((x, y), label, font=font)
        draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(0, 0, 0), outline=color, width=1)
        draw.text((x, y), label, fill=color, font=font)
    graph_y = header_height + rendered.height
    draw.line((0, graph_y, canvas_width, graph_y), fill=(75, 80, 90), width=1)
    for line_index, line in enumerate(graph_lines):
        draw.text((6, graph_y + 4 + line_index * line_height), _fit_text(line, draw, font, canvas_width - 12), fill="white", font=font)
    return canvas


def render(output: Path, config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    dependencies = [output / "input.json", output / "task_spec.json", output / "tracks.json", output / "scene_graph.json", output / "trajectory_3d.json"]
    fingerprint = stage_fingerprint(config, dependencies, "render")
    video = output / "visualization.mp4"
    sheet = output / "contact_sheet.jpg"
    if not overwrite and stage_current(output, "render", fingerprint, [video, sheet]):
        return {"status": "skipped_current", "video": str(video), "contact_sheet": str(sheet)}
    context = read_json(output / "input.json")
    task = read_json(output / "task_spec.json")
    graph = read_json(output / "scene_graph.json")
    with tempfile.TemporaryDirectory(prefix="unified_sgg_render_") as temporary_name:
        temporary = Path(temporary_name)
        rendered_frames = []
        for index in range(int(context["frame_count"])):
            frame = _draw_frame(output, index, task, graph, config.get("render"))
            path = temporary / f"frame_{index:06d}.jpg"
            frame.save(path, quality=92)
            rendered_frames.append(path)
        temporary_video = temporary / "visualization.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "6",
            "-i", str(temporary / "frame_%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(temporary_video),
        ], check=True)
        video_temporary = output / ".visualization.mp4.tmp"
        shutil.copy2(temporary_video, video_temporary)
        video_temporary.replace(video)
        count = min(12, len(rendered_frames))
        chosen = sorted(set(np.linspace(0, len(rendered_frames) - 1, count, dtype=int).tolist()))
        thumbs = [Image.open(rendered_frames[index]).convert("RGB") for index in chosen]
        thumb_width = 360
        resized = [item.resize((thumb_width, round(item.height * thumb_width / item.width)), Image.Resampling.LANCZOS) for item in thumbs]
        columns = min(4, len(resized))
        rows = (len(resized) + columns - 1) // columns
        cell_height = max(item.height for item in resized)
        contact = Image.new("RGB", (columns * thumb_width, rows * cell_height), "white")
        for position, item in enumerate(resized):
            contact.paste(item, ((position % columns) * thumb_width, (position // columns) * cell_height))
        sheet_temporary = output / ".contact_sheet.jpg.tmp"
        contact.save(sheet_temporary, format="JPEG", quality=90)
        sheet_temporary.replace(sheet)
    complete_stage(output, "render", fingerprint, {"frame_count": int(context["frame_count"])})
    update_run_report(output, "render", "success")
    return {"status": "success", "video": str(video), "contact_sheet": str(sheet)}
