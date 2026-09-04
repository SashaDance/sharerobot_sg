#!/usr/bin/env python3
"""Build a diagnostic and visual review dashboard for a completed run."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVALUATED_STATUSES = ("fully_successful", "partially_successful", "unsuccessful")
STATUS_ORDER = (*EVALUATED_STATUSES, "excluded_invalid_planning_goal", "pending")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def edge_text(edge: dict[str, Any]) -> str:
    obj = edge.get("object")
    return f"{edge['subject']} {edge['relation']}" + (f" {obj}" if obj else "")


def action_runs(frames: list[dict[str, Any]]) -> list[str]:
    by_action: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for frame in frames:
        for action in frame.get("actions", []):
            key = (action["actor"], action["action"], action["object"])
            by_action[key].append(frame["frame_index"])
    result = []
    for (actor, action, obj), indices in sorted(by_action.items()):
        spans: list[tuple[int, int]] = []
        start = previous = indices[0]
        for index in indices[1:]:
            if index != previous + 1:
                spans.append((start, previous))
                start = index
            previous = index
        spans.append((start, previous))
        ranges = ",".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)
        result.append(f"{actor} {action} {obj} [{ranges}]")
    return result


def area_instability(track: dict[str, Any]) -> float | None:
    areas = sorted(frame.get("area", 0) for frame in track.get("frames", []) if frame.get("area", 0) > 0)
    if not areas:
        return None
    median = areas[len(areas) // 2]
    return round(max(areas) / median, 2) if median else None


def scene_row(run_root: Path, item: dict[str, Any], manual: dict[str, Any]) -> dict[str, Any]:
    relative = item["relative_path"]
    scene = run_root / relative
    task = read_json(scene / "task_spec.json")
    tracks = read_json(scene / "tracks.json")
    trajectory = read_json(scene / "trajectory_3d.json")
    graph = read_json(scene / "scene_graph.json")
    frame_count = graph["frame_count"]

    track_rows = {}
    flags = {str(flag).strip().replace(" ", "-") for flag in item.get("automated_flags", [])}
    for track in tracks["tracks"]:
        visible = track.get("visible_frame_count", sum(
            frame.get("status") == "visible" for frame in track.get("frames", [])
        ))
        role = track["role"]
        instability = area_instability(track)
        track_rows[role] = {
            "visible": visible,
            "total": frame_count,
            "fallback": track.get("fallback_frame_count", 0),
            "area_instability": instability,
            "selected_prompt": track.get("sam3_selected_prompt"),
        }
        if role in {"robot", "manipulated_object"} and visible < frame_count:
            flags.add("required-mask-gap")
        if role in {"robot", "manipulated_object"} and visible < frame_count / 2:
            flags.add("insufficient-visual-evidence")
        if instability is not None and instability >= 8:
            flags.add("unstable-mask-area")

    valid_trajectory = sum(point.get("status") == "valid" for point in trajectory["points"])
    if valid_trajectory < frame_count:
        flags.add("trajectory-gap")
    if valid_trajectory < frame_count / 2:
        flags.add("trajectory-failure")

    non_robot = [
        (entity["role"], entity["canonical_name"].strip().lower())
        for entity in task["entities"] if entity["role"] != "robot"
    ]
    name_counts = Counter(name for _, name in non_robot)
    duplicated = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicated:
        flags.add("role-coreference")

    frames = graph["frames"]
    review = manual.get(relative, {"status": "pending", "categories": [], "notes": ""})
    return {
        "relative_path": relative,
        "dataset": relative.split("/", 1)[0],
        "episode": relative.rsplit("/", 1)[-1],
        "goal": item.get("planning_goal") or task.get("planning_goal", ""),
        "task_family": item.get("task_family"),
        "entities": [
            {"role": e["role"], "name": e["canonical_name"], "prompt": e["sam_prompt"]}
            for e in task["entities"]
        ],
        "tracks": track_rows,
        "trajectory_valid": valid_trajectory,
        "frame_count": frame_count,
        "initial_edges": sorted(edge_text(edge) for edge in frames[0].get("state_edges", [])),
        "final_edges": sorted(edge_text(edge) for edge in frames[-1].get("state_edges", [])),
        "action_runs": action_runs(frames),
        "flags": sorted(flags),
        "duplicate_names": duplicated,
        "status": review.get("status", "pending"),
        "categories": review.get("categories", []),
        "notes": review.get("notes", ""),
        "contact_sheet": f"../review/{relative}/contact_sheet.jpg",
        "visualization": f"../review/{relative}/visualization.mp4",
    }


def render_html(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["status"] for row in rows)
    evaluated_count = sum(counts[status] for status in EVALUATED_STATUSES)
    datasets = sorted({row["dataset"] for row in rows})
    flags = sorted({flag for row in rows for flag in row["flags"]})
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    options = "".join(f'<option value="{html.escape(x)}">{html.escape(x)}</option>' for x in datasets)
    flag_options = "".join(f'<option value="{html.escape(x)}">{html.escape(x)}</option>' for x in flags)
    count_text = (
        f"evaluated: {evaluated_count}/{len(rows)} · "
        + " · ".join(f"{key}: {counts.get(key, 0)}" for key in STATUS_ORDER)
    )
    dataset_rows = []
    for dataset in datasets:
        dataset_counts = Counter(row["status"] for row in rows if row["dataset"] == dataset)
        dataset_rows.append(
            "<tr>"
            f"<td>{html.escape(dataset)}</td>"
            f"<td>{dataset_counts.get('fully_successful', 0)}</td>"
            f"<td>{dataset_counts.get('partially_successful', 0)}</td>"
            f"<td>{dataset_counts.get('unsuccessful', 0)}</td>"
            f"<td>{dataset_counts.get('excluded_invalid_planning_goal', 0)}</td>"
            "</tr>"
        )
    category_counts = Counter(
        category
        for row in rows
        if row["status"] in EVALUATED_STATUSES
        for category in row["categories"]
    )
    category_text = " · ".join(
        f"{html.escape(category)}: {count}" for category, count in category_counts.most_common()
    ) or "No manual failure categories"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>100-scene SGG analysis</title><style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--good:#3fb950;--partial:#d29922;--bad:#f85149;--excluded:#a371f7;--pending:#8b949e}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}}header{{position:sticky;top:0;z-index:5;background:#0d1117ee;border-bottom:1px solid var(--line);padding:14px 20px}}h1{{font-size:20px;margin:0 0 8px}}.summary{{color:var(--muted);margin-bottom:10px}}.controls{{display:flex;gap:8px;flex-wrap:wrap}}select,input{{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px}}input{{min-width:280px}}.overview{{margin:18px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}}.overview table{{max-width:1050px}}.overview th{{text-align:left;padding:4px;color:var(--muted)}}.categories{{color:var(--muted);margin-top:10px}}main{{padding:0 18px 18px;display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:14px}}article{{background:var(--panel);border:1px solid var(--line);border-left:5px solid var(--pending);border-radius:8px;overflow:hidden}}article.fully_successful{{border-left-color:var(--good)}}article.partially_successful{{border-left-color:var(--partial)}}article.unsuccessful{{border-left-color:var(--bad)}}article.excluded_invalid_planning_goal{{border-left-color:var(--excluded)}}.head{{padding:12px}}h2{{font-size:15px;margin:0 0 6px;word-break:break-word}}.goal{{font-weight:600}}.meta,.notes{{color:var(--muted)}}.badges{{display:flex;gap:5px;flex-wrap:wrap}}.badge{{background:#21262d;border:1px solid var(--line);padding:2px 6px;border-radius:999px;font-size:12px}}video,img{{display:block;width:100%;background:#000}}details{{border-top:1px solid var(--line);padding:8px 12px}}summary{{cursor:pointer}}table{{width:100%;border-collapse:collapse;margin-top:6px}}td{{vertical-align:top;border-top:1px solid var(--line);padding:4px}}td:first-child{{color:var(--muted)}}a{{color:#58a6ff}}.hidden{{display:none}}
</style></head><body><header><h1>100-scene SGG semantic review</h1><div class="summary">{html.escape(count_text)}</div><div class="controls"><select id="dataset"><option value="">All datasets</option>{options}</select><select id="status"><option value="">All statuses</option><option>fully_successful</option><option>partially_successful</option><option>unsuccessful</option><option>excluded_invalid_planning_goal</option><option>pending</option></select><select id="flag"><option value="">All diagnostic flags</option>{flag_options}</select><input id="search" placeholder="Search goal, episode, entity, note"></div></header><section class="overview"><b>Manual outcome by dataset</b><table><thead><tr><th>Dataset</th><th>Fully successful</th><th>Partially successful</th><th>Unsuccessful</th><th>Excluded: invalid goal</th></tr></thead><tbody>{''.join(dataset_rows)}</tbody></table><div class="categories"><b>Failure categories (evaluated scenes only):</b> {category_text}</div></section><main id="cards"></main>
<script>const rows={data};const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
function card(r){{const entities=r.entities.map(e=>`<tr><td>${{esc(e.role)}}</td><td>${{esc(e.name)}}<br><span class=meta>SAM: ${{esc(e.prompt)}}</span></td></tr>`).join('');const tracks=Object.entries(r.tracks).map(([k,v])=>`${{k}} ${{v.visible}}/${{v.total}}${{v.fallback?' +'+v.fallback+' repair':''}}`).join(' · ');return `<article class="${{esc(r.status)}}" data-dataset="${{esc(r.dataset)}}" data-status="${{esc(r.status)}}" data-flags="${{esc(r.flags.join(' '))}}"><div class=head><h2>${{esc(r.relative_path)}}</h2><div class=goal>${{esc(r.goal)}}</div><div class=meta>${{esc(r.task_family||'unclassified')}} · trajectory ${{r.trajectory_valid}}/${{r.frame_count}} · ${{esc(tracks)}}</div><div class=badges>${{r.flags.map(x=>`<span class=badge>${{esc(x)}}</span>`).join('')}}</div><div class=notes><b>${{esc(r.status)}}</b>${{r.categories.length?' · '+esc(r.categories.join(', ')):''}}${{r.notes?' — '+esc(r.notes):''}}</div></div><video controls preload=none poster="${{esc(r.contact_sheet)}}" src="${{esc(r.visualization)}}"></video><details><summary>Graph and prompts</summary><table>${{entities}}<tr><td>Initial</td><td>${{esc(r.initial_edges.join('; ')||'none')}}</td></tr><tr><td>Final</td><td>${{esc(r.final_edges.join('; ')||'none')}}</td></tr><tr><td>Actions</td><td>${{esc(r.action_runs.join('; ')||'none')}}</td></tr></table><p><a href="${{esc(r.contact_sheet)}}">Open full contact sheet</a></p></details></article>`}}
function render(){{const d=document.querySelector('#dataset').value,s=document.querySelector('#status').value,f=document.querySelector('#flag').value,q=document.querySelector('#search').value.toLowerCase();document.querySelector('#cards').innerHTML=rows.filter(r=>(!d||r.dataset===d)&&(!s||r.status===s)&&(!f||r.flags.includes(f))&&(!q||JSON.stringify(r).toLowerCase().includes(q))).map(card).join('')}}document.querySelectorAll('select,input').forEach(x=>x.addEventListener('input',render));render();</script></body></html>"""


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, dict[str, int]] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        counts = Counter(row["status"] for row in rows if row["dataset"] == dataset)
        datasets[dataset] = {
            status: counts.get(status, 0)
            for status in STATUS_ORDER
        }
    evaluated_rows = [row for row in rows if row["status"] in EVALUATED_STATUSES]
    excluded_rows = [row for row in rows if row["status"] == "excluded_invalid_planning_goal"]
    return {
        "scene_count": len(rows),
        "evaluated_scene_count": len(evaluated_rows),
        "excluded_scene_count": len(excluded_rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "dataset_status_counts": datasets,
        "manual_failure_categories": dict(Counter(
            category for row in evaluated_rows for category in row["categories"]
        )),
        "manual_exclusion_categories": dict(Counter(
            category for row in excluded_rows for category in row["categories"]
        )),
        "automated_diagnostic_flags": dict(Counter(
            flag for row in rows for flag in row["flags"]
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-reviews", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = read_json(args.run_root / "review_index.json")
    manual = read_json(args.manual_reviews) if args.manual_reviews and args.manual_reviews.is_file() else {}
    rows = [scene_row(args.run_root, item, manual) for item in index["scenes"]]
    summary = build_summary(rows)
    (args.output_dir / "scene_analysis.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "scene_analysis.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "dataset", "episode", "goal", "task_family", "status", "trajectory_valid", "frame_count", "flags", "categories", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "; ".join(row[key]) if isinstance(row.get(key), list) else row.get(key) for key in writer.fieldnames})
    (args.output_dir / "index.html").write_text(render_html(rows), encoding="utf-8")
    print(json.dumps({**summary, "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
