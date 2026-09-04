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

STATE_COLOR = (78, 205, 196)
ACTION_COLOR = (255, 183, 77)
NEW_COLOR = (117, 226, 126)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Use a readable bundled system font when available, with a safe fallback."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


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


def _entity_label(entity_id: str | None, task: dict[str, Any]) -> str:
    if not entity_id:
        return ""
    entity = next(
        (item for item in task.get("entities", []) if item.get("entity_id") == entity_id),
        None,
    )
    if not entity:
        return str(entity_id)
    marker = ROLE_MARKERS.get(entity.get("role", ""), "?")
    name = entity.get("canonical_name") or entity_id
    return f"{marker}  {name}"


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("subject") or ""),
        str(edge.get("relation") or ""),
        str(edge.get("object") or ""),
    )


def _action_key(action: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(action.get("actor") or ""),
        str(action.get("action") or ""),
        str(action.get("object") or ""),
    )


def _relation_rows(
    row: dict[str, Any],
    previous_row: dict[str, Any] | None,
    task: dict[str, Any],
) -> list[dict[str, Any]]:
    previous_edges = {
        _edge_key(item) for item in (previous_row or {}).get("state_edges", [])
    }
    previous_actions = {
        _action_key(item) for item in (previous_row or {}).get("actions", [])
    }
    rows: list[dict[str, Any]] = []
    state_edges = row.get("state_edges", [])
    actions = row.get("actions", [])
    if state_edges:
        for edge in state_edges:
            subject = _entity_label(edge.get("subject"), task)
            relation = str(edge.get("relation") or "relation").upper()
            object_label = _entity_label(edge.get("object"), task)
            text = f"{subject}   — {relation} →   {object_label}" if object_label else f"{subject}   — {relation}"
            rows.append({
                "kind": "STATE",
                "text": text,
                "color": STATE_COLOR,
                "new": _edge_key(edge) not in previous_edges,
            })
    else:
        rows.append({"kind": "STATE", "text": "No state relation predicted", "color": STATE_COLOR, "new": False})
    if actions:
        for action in actions:
            actor = _entity_label(action.get("actor"), task)
            action_name = str(action.get("action") or "action").upper()
            object_label = _entity_label(action.get("object"), task)
            text = f"{actor}   — {action_name} →   {object_label}" if object_label else f"{actor}   — {action_name}"
            rows.append({
                "kind": "ACTION",
                "text": text,
                "color": ACTION_COLOR,
                "new": _action_key(action) not in previous_actions,
            })
    else:
        rows.append({"kind": "ACTION", "text": "No action predicted", "color": ACTION_COLOR, "new": False})
    return rows


def _draw_frame(
    output: Path,
    index: int,
    task: dict[str, Any],
    graph: dict[str, Any],
    render_config: dict[str, Any] | None = None,
    tracks_by_entity: dict[str, dict[str, Any]] | None = None,
    relation_row_count: int | None = None,
) -> Image.Image:
    render_config = render_config or {}
    tracks_by_entity = tracks_by_entity or {}
    mask_alpha = float(render_config.get("mask_alpha", 0.48))
    boundary_width = int(render_config.get("boundary_width", 2))
    min_width = int(render_config.get("min_width", 640))
    image = Image.open(output / "frames" / f"frame_{index:06d}.png").convert("RGB")
    pixels = np.asarray(image).copy()
    labels = []
    grounding_boxes = []
    visible_entities: set[str] = set()
    for entity in task["entities"]:
        color_tuple = ROLE_COLORS.get(entity["role"], (210, 210, 210))
        grounding = entity.get("grounding")
        frame_grounding = next(
            (
                row["bbox_xyxy_1000"]
                for row in entity.get("frame_grounding", [])
                if int(row["frame_index"]) == index
            ),
            None,
        )
        active_box = (
            frame_grounding
            if frame_grounding is not None
            else grounding["bbox_xyxy_1000"]
            if grounding and int(grounding["frame_index"]) == index
            else None
        )
        if active_box is not None:
            x_min, y_min, x_max, y_max = active_box
            grounding_boxes.append((
                round(x_min * image.width / 1000),
                round(y_min * image.height / 1000),
                round(x_max * image.width / 1000),
                round(y_max * image.height / 1000),
                color_tuple,
            ))
        mask = np.asarray(Image.open(output / "masks" / entity["entity_id"] / f"frame_{index:06d}.png").convert("L")) > 0
        if not mask.any():
            continue
        visible_entities.add(entity["entity_id"])
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
        grounding_boxes = [
            (round(x_min * scale), round(y_min * scale), round(x_max * scale), round(y_max * scale), color)
            for x_min, y_min, x_max, y_max, color in grounding_boxes
        ]

    header_font = _load_font(14)
    header_bold_font = _load_font(14, bold=True)
    marker_font = _load_font(15, bold=True)
    line_height = 20
    header_lines = [f"Goal: {task.get('planning_goal', '')}"]
    for entity in task["entities"]:
        marker = ROLE_MARKERS.get(entity["role"], "?")
        status = "visible" if entity["entity_id"] in visible_entities else "not visible"
        grounding = entity.get("grounding")
        track = tracks_by_entity.get(entity["entity_id"], {})
        prompt = (
            (
                f"SAM3 core: \"{track.get('core_concept')}\"; broad: "
                f"\"{track.get('broad_concept')}\"; selected: "
                f"{track.get('selected_concept')}/{track.get('selected_candidate_id')}"
            )
            if track.get("selection_method") == "agent_sam3_tracks_qwen_visual_pruning_no_boxes"
            else str(render_config["grounding_label"])
            if render_config.get("grounding_label")
            else f"SAM3 text: \"{entity['sam_prompt']}\" + Qwen all-frame verifier"
            if entity.get("frame_grounding") is not None
            else f"SAM box: frame {grounding['frame_index']} {grounding['bbox_xyxy_1000']}"
            if grounding else f"SAM3 prompt: \"{entity['sam_prompt']}\""
        )
        header_lines.append(
            f"{marker} {entity['role']} | canonical: {entity['canonical_name']} | {prompt} | {status}"
        )
    header_height = 10 + line_height * len(header_lines)

    row = graph["frames"][index]
    previous_row = graph["frames"][index - 1] if index > 0 else None
    relation_rows = _relation_rows(row, previous_row, task)
    relation_row_count = max(relation_row_count or len(relation_rows), len(relation_rows))
    relation_font_size = max(18, min(24, rendered.width // 30))
    relation_font = _load_font(relation_font_size, bold=True)
    badge_font = _load_font(max(13, relation_font_size - 6), bold=True)
    relation_line_height = relation_font_size + 16
    graph_height = 48 + relation_line_height * relation_row_count

    canvas_width = rendered.width + (rendered.width % 2)
    canvas_height = header_height + rendered.height + graph_height
    canvas_height += canvas_height % 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (16, 18, 22))
    canvas.paste(rendered, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    for x_min, y_min, x_max, y_max, color in grounding_boxes:
        draw.rectangle(
            (x_min, y_min + header_height, x_max, y_max + header_height),
            outline=color,
            width=max(2, boundary_width),
        )
    draw.line((0, header_height - 1, canvas_width, header_height - 1), fill=(75, 80, 90), width=1)
    for line_index, line in enumerate(header_lines):
        y = 5 + line_index * line_height
        if line_index == 0:
            color = (245, 245, 245)
        else:
            entity = task["entities"][line_index - 1]
            color = ROLE_COLORS.get(entity["role"], (210, 210, 210))
        selected_font = header_bold_font if line_index == 0 else header_font
        draw.text((8, y), _fit_text(line, draw, selected_font, canvas_width - 16), fill=color, font=selected_font)
    for x, y, label, color in labels:
        x = min(x, canvas_width - 16)
        y = min(y + header_height, header_height + rendered.height - line_height)
        box = draw.textbbox((x, y), label, font=marker_font)
        draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(0, 0, 0), outline=color, width=1)
        draw.text((x, y), label, fill=color, font=marker_font)
    graph_y = header_height + rendered.height
    draw.rectangle((0, graph_y, canvas_width, canvas_height), fill=(9, 13, 19))
    draw.line((0, graph_y, canvas_width, graph_y), fill=(105, 115, 130), width=2)
    title_font = _load_font(17, bold=True)
    draw.text(
        (10, graph_y + 8),
        f"PREDICTED RELATIONS  •  FRAME {index + 1}/{len(graph['frames'])}",
        fill=(245, 247, 250),
        font=title_font,
    )
    for row_index, relation_row in enumerate(relation_rows):
        y_min = graph_y + 38 + row_index * relation_line_height
        y_max = y_min + relation_line_height - 6
        color = relation_row["color"]
        outline = NEW_COLOR if relation_row["new"] else tuple(max(25, int(channel * 0.62)) for channel in color)
        draw.rounded_rectangle(
            (8, y_min, canvas_width - 8, y_max),
            radius=7,
            fill=(20, 27, 36),
            outline=outline,
            width=3 if relation_row["new"] else 1,
        )
        badge_width = 72
        draw.rounded_rectangle(
            (15, y_min + 6, 15 + badge_width, y_max - 6),
            radius=5,
            fill=tuple(max(0, int(channel * 0.35)) for channel in color),
        )
        draw.text((24, y_min + 9), relation_row["kind"], fill=color, font=badge_font)
        text_x = 99
        new_label_width = 0
        if relation_row["new"]:
            new_text = "NEW"
            new_box = draw.textbbox((0, 0), new_text, font=badge_font)
            new_label_width = new_box[2] - new_box[0] + 18
            draw.rounded_rectangle(
                (canvas_width - new_label_width - 15, y_min + 6, canvas_width - 15, y_max - 6),
                radius=5,
                fill=(31, 87, 47),
            )
            draw.text((canvas_width - new_label_width - 6, y_min + 9), new_text, fill=NEW_COLOR, font=badge_font)
        available_width = canvas_width - text_x - 18 - new_label_width
        relation_text = _fit_text(relation_row["text"], draw, relation_font, available_width)
        draw.text((text_x, y_min + 6), relation_text, fill=(242, 245, 248), font=relation_font)
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
    tracks = read_json(output / "tracks.json")
    tracks_by_entity = {
        track["entity_id"]: track for track in tracks.get("tracks", [])
    }
    relation_row_count = max(
        len(_relation_rows(row, graph["frames"][index - 1] if index > 0 else None, task))
        for index, row in enumerate(graph["frames"])
    )
    with tempfile.TemporaryDirectory(prefix="unified_sgg_render_") as temporary_name:
        temporary = Path(temporary_name)
        rendered_frames = []
        for index in range(int(context["frame_count"])):
            frame = _draw_frame(
                output, index, task, graph, config.get("render"), tracks_by_entity,
                relation_row_count,
            )
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
