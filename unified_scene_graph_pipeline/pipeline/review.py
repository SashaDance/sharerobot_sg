from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from batch import entries, paths
from common import read_json, utc_now, write_json_atomic


FAILURE_CATEGORIES = (
    "api/schema", "wrong role", "wrong SAM instance", "lost track",
    "incorrect relation/action", "DA3 failure", "trajectory failure",
)


def generate_review_index(manifest: Path, source_root: Path, output_root: Path) -> dict[str, Any]:
    rows = []
    for item in entries(manifest):
        _, scene = paths(item, source_root, output_root)
        flags = []
        validation_path = scene / "validation_report.json"
        validation = read_json(validation_path) if validation_path.is_file() else None
        if validation is None or validation.get("status") != "passed":
            flags.append("api/schema")
        tracks_path = scene / "tracks.json"
        if tracks_path.is_file():
            tracks = read_json(tracks_path)
            for track in tracks.get("tracks", []):
                visible = sum(frame.get("status") == "visible" for frame in track.get("frames", []))
                if visible == 0:
                    flags.append("lost track")
        trajectory_path = scene / "trajectory_3d.json"
        if trajectory_path.is_file():
            trajectory = read_json(trajectory_path)
            valid = sum(point.get("status") == "valid" for point in trajectory.get("points", []))
            if valid < max(1, len(trajectory.get("points", [])) // 2):
                flags.append("trajectory failure")
        elif (scene / "da3").exists():
            flags.append("trajectory failure")
        if not (scene / "da3" / "camera_poses.txt").is_file():
            flags.append("DA3 failure")
        graph_path = scene / "scene_graph.json"
        if graph_path.is_file():
            graph = read_json(graph_path)
            graph_frames = graph.get("frames", [])
            has_graph_content = any(row.get("state_edges") or row.get("actions") for row in graph_frames)
            has_actions = any(row.get("actions") for row in graph_frames)
            task_path = scene / "task_spec.json"
            task = read_json(task_path) if task_path.is_file() else {}
            if not has_graph_content or (task.get("task_actions") and not has_actions):
                flags.append("incorrect relation/action")
        rows.append({
            "relative_path": str(scene.relative_to(output_root)),
            "planning_goal": item.get("planning_goal"),
            "task_family": item.get("task_family"),
            "automated_flags": sorted(set(flags)),
            "contact_sheet": str((scene / "contact_sheet.jpg").relative_to(output_root)) if (scene / "contact_sheet.jpg").is_file() else None,
            "visualization": str((scene / "visualization.mp4").relative_to(output_root)) if (scene / "visualization.mp4").is_file() else None,
            "manual_review": {"status": "pending", "failure_categories": [], "notes": ""},
        })
    report = {
        "schema_version": "unified_sgg_review_index_v1", "generated_at": utc_now(),
        "failure_categories": list(FAILURE_CATEGORIES), "scene_count": len(rows), "scenes": rows,
    }
    write_json_atomic(output_root / "review_index.json", report)
    cards = []
    for row in rows:
        media = ""
        if row["contact_sheet"]:
            media += f'<a href="{html.escape(row["visualization"] or row["contact_sheet"])}"><img loading="lazy" src="{html.escape(row["contact_sheet"])}"></a>'
        cards.append(
            '<article><h2>' + html.escape(row["relative_path"]) + '</h2>'
            + '<p>' + html.escape(row.get("planning_goal") or "") + '</p>'
            + '<p><b>Automated flags:</b> ' + html.escape(", ".join(row["automated_flags"]) or "none") + '</p>'
            + media + '<p><b>Manual checklist:</b> object identity · mask tracking · initial/final relations · action timing · depth · 3D trajectory</p></article>'
        )
    document = """<!doctype html><meta charset=utf-8><title>Unified SGG review</title><style>
body{font:14px system-ui;margin:20px;background:#f4f5f7}article{background:white;padding:16px;margin:16px 0;border-radius:8px}img{max-width:100%;height:auto}h2{font-size:17px}
</style><h1>Unified SGG review index</h1>""" + "\n".join(cards)
    temporary = output_root / ".review_index.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output_root / "review_index.html")
    return report
