from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import (
    complete_stage,
    image_paths,
    mask_label_placement,
    read_json,
    stage_current,
    stage_fingerprint,
    update_run_report,
)


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


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for directory in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
    ):
        candidate = directory / name
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _mask_boundary(mask: np.ndarray, width: int) -> np.ndarray:
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


def _fit_text(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    while text and draw.textlength(text + suffix, font=font) > max_width:
        text = text[:-1]
    return text + suffix


def _track_map(tracks: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(track["entity_id"]): track
        for track in tracks.get("tracks", [])
        if isinstance(track, dict) and track.get("entity_id")
    }


def _frame_status(track: dict[str, Any], frame_index: int) -> str:
    for row in track.get("frames", []):
        if int(row.get("frame_index", -1)) == frame_index:
            return str(row.get("status", "unknown"))
    return "unknown"


def _grounding_box(entity: dict[str, Any], frame_index: int) -> list[int] | None:
    for row in entity.get("frame_grounding") or []:
        if int(row.get("frame_index", -1)) == frame_index:
            return row.get("bbox_xyxy_1000")
    return None


def draw_frame(
    output: Path,
    frame_index: int,
    task: dict[str, Any],
    tracks: dict[str, Any],
    config: dict[str, Any],
) -> Image.Image:
    frame_paths = image_paths(output / "frames")
    if not 0 <= frame_index < len(frame_paths):
        raise IndexError(frame_index)
    image = Image.open(frame_paths[frame_index]).convert("RGB")
    pixels = np.asarray(image).copy()
    height, width = pixels.shape[:2]
    visualize_config = config.get("visualize", {})
    alpha = float(visualize_config.get("mask_alpha", 0.45))
    boundary_width = int(visualize_config.get("boundary_width", 3))
    min_width = int(visualize_config.get("min_width", 640))
    show_boxes = bool(visualize_config.get("show_grounding_boxes", False))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("visualize.mask_alpha must be between 0 and 1")
    if boundary_width < 1:
        raise ValueError("visualize.boundary_width must be positive")
    if min_width < 1:
        raise ValueError("visualize.min_width must be positive")

    tracks_by_entity = _track_map(tracks)
    labels: list[tuple[int, int, float, str, tuple[int, int, int]]] = []
    boxes: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
    status_rows: list[tuple[dict[str, Any], str]] = []

    for entity in task.get("entities", []):
        entity_id = str(entity["entity_id"])
        role = str(entity.get("role", entity_id))
        color_tuple = ROLE_COLORS.get(role, (210, 210, 210))
        mask_path = output / "masks" / entity_id / f"frame_{frame_index:06d}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != (height, width):
            raise ValueError(
                f"Mask dimensions {mask.shape} do not match frame {(height, width)}: {mask_path}"
            )
        track = tracks_by_entity.get(entity_id, {})
        status = _frame_status(track, frame_index)
        status_rows.append((entity, status))
        if mask.any():
            color = np.asarray(color_tuple, dtype=np.float32)
            pixels[mask] = (
                (1.0 - alpha) * pixels[mask].astype(np.float32) + alpha * color
            ).astype(np.uint8)
            pixels[_mask_boundary(mask, boundary_width)] = np.asarray(
                color_tuple, dtype=np.uint8
            )
            x, y, radius = mask_label_placement(mask)
            labels.append((x, y, radius, ROLE_MARKERS.get(role, "?"), color_tuple))
        if show_boxes:
            box = _grounding_box(entity, frame_index)
            if box is not None:
                x1, y1, x2, y2 = box
                boxes.append((
                    round(x1 * width / 1000),
                    round(y1 * height / 1000),
                    round(x2 * width / 1000),
                    round(y2 * height / 1000),
                    color_tuple,
                ))

    rendered = Image.fromarray(pixels)
    scale = max(1.0, min_width / rendered.width)
    if scale > 1.0:
        rendered = rendered.resize(
            (round(rendered.width * scale), round(rendered.height * scale)),
            Image.Resampling.LANCZOS,
        )
        labels = [
            (round(x * scale), round(y * scale), radius * scale, marker, color)
            for x, y, radius, marker, color in labels
        ]
        boxes = [
            (
                round(x1 * scale), round(y1 * scale),
                round(x2 * scale), round(y2 * scale), color,
            )
            for x1, y1, x2, y2, color in boxes
        ]

    font = _load_font(14)
    bold_font = _load_font(14, bold=True)
    line_height = 21
    header_lines: list[tuple[str, tuple[int, int, int], bool]] = [(
        f"Goal: {task.get('planning_goal', '')}", (245, 245, 245), True,
    )]
    for entity, status in status_rows:
        role = str(entity.get("role", entity["entity_id"]))
        marker = ROLE_MARKERS.get(role, "?")
        canonical_name = str(entity.get("canonical_name", entity["entity_id"]))
        sam_prompt = str(entity.get("sam_prompt", ""))
        header_lines.append((
            f'{marker} {role} | canonical: {canonical_name} | SAM prompt: "{sam_prompt}" | {status}',
            ROLE_COLORS.get(role, (210, 210, 210)),
            False,
        ))
    header_height = 10 + line_height * len(header_lines)
    footer_height = 30
    canvas_width = rendered.width + rendered.width % 2
    canvas_height = header_height + rendered.height + footer_height
    canvas_height += canvas_height % 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (16, 18, 22))
    canvas.paste(rendered, (0, header_height))
    draw = ImageDraw.Draw(canvas)

    for line_index, (line, color, bold) in enumerate(header_lines):
        selected_font = bold_font if bold else font
        draw.text(
            (8, 5 + line_index * line_height),
            _fit_text(line, draw, selected_font, canvas_width - 16),
            fill=color,
            font=selected_font,
        )
    draw.line(
        (0, header_height - 1, canvas_width, header_height - 1),
        fill=(75, 80, 90),
    )

    for x1, y1, x2, y2, color in boxes:
        draw.rectangle(
            (x1, y1 + header_height, x2, y2 + header_height),
            outline=color,
            width=max(2, boundary_width),
        )
    for x, y, radius, marker, color in labels:
        font_size = max(12, min(40, round(radius * 1.6)))
        marker_font = _load_font(font_size, bold=True)
        box = draw.textbbox((0, 0), marker, font=marker_font, stroke_width=1)
        label_width = box[2] - box[0]
        label_height = box[3] - box[1]
        draw.text(
            (
                max(2, min(x - label_width // 2, canvas_width - label_width - 2)),
                max(
                    header_height,
                    min(
                        y + header_height - label_height // 2,
                        header_height + rendered.height - label_height - 2,
                    ),
                ),
            ),
            marker,
            fill=color,
            font=marker_font,
            stroke_width=max(1, font_size // 10),
            stroke_fill=(0, 0, 0),
        )

    footer_y = header_height + rendered.height
    draw.rectangle((0, footer_y, canvas_width, canvas_height), fill=(9, 13, 19))
    draw.text(
        (8, footer_y + 6),
        f"FRAME {frame_index + 1}/{len(frame_paths)}",
        fill=(235, 238, 242),
        font=bold_font,
    )
    return canvas


def visualize(
    output: Path,
    config: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    dependencies = [
        output / "input.json",
        output / "task_spec.json",
        output / "tracks.json",
        output / "frames",
        output / "masks",
    ]
    fingerprint = stage_fingerprint(config, dependencies, "visualize")
    video = output / "visualization.mp4"
    contact_sheet = output / "contact_sheet.jpg"
    if not overwrite and stage_current(
        output, "visualize", fingerprint, [video, contact_sheet]
    ):
        return {
            "status": "skipped_current",
            "video": str(video),
            "contact_sheet": str(contact_sheet),
        }

    context = read_json(output / "input.json")
    task = read_json(output / "task_spec.json")
    tracks = read_json(output / "tracks.json")
    frame_count = int(context["frame_count"])
    if len(image_paths(output / "frames")) != frame_count:
        raise ValueError("Prepared frame count changed before visualization")
    visualize_config = config.get("visualize", {})
    fps = int(visualize_config.get("fps", 6))
    sheet_frame_count = int(visualize_config.get("contact_sheet_frames", 6))
    sheet_columns = int(visualize_config.get("contact_sheet_columns", 2))
    thumb_width = int(visualize_config.get("contact_sheet_thumb_width", 640))
    if min(fps, sheet_frame_count, sheet_columns, thumb_width) < 1:
        raise ValueError("Visualization FPS and contact-sheet dimensions must be positive")

    try:
        with tempfile.TemporaryDirectory(prefix="segmentation_visualize_") as name:
            temporary = Path(name)
            rendered_paths: list[Path] = []
            for frame_index in range(frame_count):
                frame = draw_frame(output, frame_index, task, tracks, config)
                path = temporary / f"frame_{frame_index:06d}.jpg"
                frame.save(path, format="JPEG", quality=92)
                rendered_paths.append(path)

            temporary_video = temporary / "visualization.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", str(fps),
                    "-i", str(temporary / "frame_%06d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(temporary_video),
                ],
                check=True,
            )
            video_temporary = output / ".visualization.mp4.tmp"
            shutil.copy2(temporary_video, video_temporary)
            video_temporary.replace(video)

            count = min(sheet_frame_count, len(rendered_paths))
            chosen = sorted(set(
                np.linspace(0, len(rendered_paths) - 1, count, dtype=int).tolist()
            ))
            thumbnails = [
                Image.open(rendered_paths[index]).convert("RGB") for index in chosen
            ]
            resized = [
                item.resize(
                    (thumb_width, round(item.height * thumb_width / item.width)),
                    Image.Resampling.LANCZOS,
                )
                for item in thumbnails
            ]
            columns = min(sheet_columns, len(resized))
            rows = (len(resized) + columns - 1) // columns
            cell_height = max(item.height for item in resized)
            sheet = Image.new(
                "RGB", (columns * thumb_width, rows * cell_height), (16, 18, 22)
            )
            for position, item in enumerate(resized):
                sheet.paste(
                    item,
                    (
                        (position % columns) * thumb_width,
                        (position // columns) * cell_height,
                    ),
                )
            sheet_temporary = output / ".contact_sheet.jpg.tmp"
            sheet.save(sheet_temporary, format="JPEG", quality=90)
            sheet_temporary.replace(contact_sheet)
    except Exception as error:
        update_run_report(
            output,
            "visualize",
            "failed",
            {"error": f"{type(error).__name__}: {error}"},
        )
        raise

    complete_stage(output, "visualize", fingerprint, {"frame_count": frame_count})
    update_run_report(output, "visualize", "success", {"frame_count": frame_count})
    return {
        "status": "success",
        "video": str(video),
        "contact_sheet": str(contact_sheet),
    }
