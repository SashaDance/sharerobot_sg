#!/usr/bin/env python3
"""Deterministically select the 100-scene validation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path


SEED = 20260901
FAMILIES = ("pick_place", "stack", "insert", "pour", "push", "open_close", "deformable")


def task_family(goal: str) -> str:
    text = goal.lower()
    if re.search(r"\b(open|close|closing|opening|drawer|door)\b", text):
        return "open_close"
    if re.search(r"\bpour", text):
        return "pour"
    if re.search(r"\b(push|slide|pull|drag)", text):
        return "push"
    if re.search(r"\bstack|on top of|onto another", text):
        return "stack"
    if re.search(r"\b(insert|slot|hole|peg|compartment)", text):
        return "insert"
    if re.search(r"\b(cloth|towel|rag|fabric|sponge|wipe|fold|deform)", text):
        return "deformable"
    return "pick_place"


def png_resolution(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def rank(item: dict) -> str:
    return hashlib.sha256(f"{SEED}:{item['relative_path']}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    source = json.loads((root / "manifest.json").read_text())
    items = source["episodes"]
    enriched = []
    for original in items:
        item = dict(original)
        item["task_family"] = task_family(item["planning_goal"])
        item["resolution"] = list(png_resolution(root / item["relative_path"] / "images" / "frame_0.png"))
        enriched.append(item)
    datasets = sorted({item["dataset_name"] for item in enriched})
    quotas = {dataset: 100 // len(datasets) for dataset in datasets}
    for dataset in datasets[: 100 % len(datasets)]:
        quotas[dataset] += 1
    selected: dict[str, dict] = {}
    counts = Counter()

    def choose(candidates: list[dict]) -> None:
        candidates = [item for item in candidates if item["relative_path"] not in selected]
        if not candidates:
            raise RuntimeError("Cannot satisfy a validation stratum")
        item = min(candidates, key=lambda row: (counts[row["dataset_name"]] / quotas[row["dataset_name"]], rank(row)))
        selected[item["relative_path"]] = item
        counts[item["dataset_name"]] += 1

    for family in FAMILIES:
        choose([item for item in enriched if item["task_family"] == family])
    for resolution in sorted({tuple(item["resolution"]) for item in enriched}):
        if not any(tuple(item["resolution"]) == resolution for item in selected.values()):
            choose([item for item in enriched if tuple(item["resolution"]) == resolution])
    for dataset in datasets:
        candidates = sorted((item for item in enriched if item["dataset_name"] == dataset), key=rank)
        for item in candidates:
            if counts[dataset] >= quotas[dataset]:
                break
            if item["relative_path"] not in selected:
                selected[item["relative_path"]] = item
                counts[dataset] += 1
    if len(selected) != 100:
        raise RuntimeError(f"Expected 100 selections, got {len(selected)}")
    selected_items = sorted(selected.values(), key=lambda item: (item["dataset_name"], rank(item)))
    result = {
        "schema_version": "unified_sgg_selection_v1",
        "name": "validation_100_stratified",
        "seed": SEED,
        "selection": "dataset quotas plus forced task-family and image-resolution coverage; SHA-256 seeded ordering",
        "dataset_counts": dict(sorted(Counter(item["dataset_name"] for item in selected_items).items())),
        "task_family_counts": dict(sorted(Counter(item["task_family"] for item in selected_items).items())),
        "resolution_counts": {f"{key[0]}x{key[1]}": value for key, value in sorted(Counter(tuple(item["resolution"]) for item in selected_items).items())},
        "episodes": selected_items,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
