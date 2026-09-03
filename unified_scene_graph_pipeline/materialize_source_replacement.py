#!/usr/bin/env python3
"""Retain a selected dataset and replace one excluded source deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalized_goal(value: str) -> str:
    value = re.sub(r"[^\w\s]", " ", value.lower().strip())
    return re.sub(r"\s+", " ", value).strip()


def deterministic_rank(episode_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--pool-work", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exclude-source", required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    base_manifest = read_json(args.base_root / "manifest.json")
    proposal_manifest = read_json(args.proposal_root / "manifest.json")
    proposal_report = read_json(args.proposal_root / "selection_report.json")
    quotas = {name: int(count) for name, count in proposal_report["source_quotas"].items()}

    retained = [
        dict(record) for record in base_manifest["episodes"]
        if record["dataset_name"] != args.exclude_source
    ]
    retained_ids = {record["episode_id"] for record in retained}
    if len(retained_ids) != len(retained):
        raise ValueError("Base manifest contains duplicate retained episode IDs")
    retained_counts = Counter(record["dataset_name"] for record in retained)
    if set(retained_counts) != set(quotas):
        raise ValueError(
            f"Retained sources {sorted(retained_counts)} do not match quotas {sorted(quotas)}"
        )

    pool_manifest = [
        json.loads(line)
        for line in (args.pool_work / "candidate_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    visual = np.load(args.pool_work / "visual_embeddings.npz")
    pool_ids = [record["episode_id"] for record in pool_manifest]
    if [str(value) for value in visual["episode_ids"]] != pool_ids:
        raise ValueError("Pool embeddings do not match the candidate manifest")
    signature_by_id = {
        episode_id: str(signature)
        for episode_id, signature in zip(pool_ids, visual["signatures"], strict=True)
    }

    proposal_candidates: dict[str, list[dict[str, Any]]] = {}
    for source in quotas:
        proposal_candidates[source] = [
            dict(record) for record in proposal_manifest["episodes"]
            if record["dataset_name"] == source and record["episode_id"] not in retained_ids
        ]

    goal_counts = Counter(normalized_goal(record["planning_goal"]) for record in retained)
    used_signatures = {
        signature_by_id[record["episode_id"]]
        for record in retained
        if record["episode_id"] in signature_by_id
    }
    additions: list[dict[str, Any]] = []
    for source in sorted(quotas):
        needed = quotas[source] - retained_counts[source]
        candidates = proposal_candidates[source]
        if needed < 0:
            raise ValueError(f"Quota for {source} is below its retained count")
        if len(candidates) < needed:
            raise ValueError(
                f"Need {needed} additions for {source}, but only {len(candidates)} proposals exist"
            )
        for _ in range(needed):
            chosen = min(
                candidates,
                key=lambda record: (
                    signature_by_id.get(record["episode_id"]) in used_signatures,
                    goal_counts[normalized_goal(record["planning_goal"])],
                    deterministic_rank(record["episode_id"], args.seed),
                ),
            )
            candidates.remove(chosen)
            additions.append(chosen)
            goal_counts[normalized_goal(chosen["planning_goal"])] += 1
            signature = signature_by_id.get(chosen["episode_id"])
            if signature is not None:
                used_signatures.add(signature)

    final_records = []
    for record in retained:
        record["selection_stage"] = "retained_after_source_exclusion"
        final_records.append(record)
    for record in additions:
        record["selection_stage"] = "replacement_multimodal_diversity"
        final_records.append(record)
    final_records.sort(key=lambda record: record["episode_id"])

    for record in final_records:
        source_root = args.base_root if record["episode_id"] in retained_ids else args.proposal_root
        source = source_root / record["relative_path"]
        destination = args.output_root / record["relative_path"]
        shutil.copytree(source, destination, copy_function=hardlink_or_copy)

    final_counts = Counter(record["dataset_name"] for record in final_records)
    if len(final_records) != sum(quotas.values()) or dict(final_counts) != quotas:
        raise RuntimeError(
            f"Final counts do not match quotas: {dict(final_counts)} versus {quotas}"
        )

    manifest = {
        **{key: value for key, value in base_manifest.items() if key != "episodes"},
        "episode_count": len(final_records),
        "total_frames": len(final_records) * int(base_manifest["frames_per_episode"]),
        "included_sources": sorted(quotas),
        "excluded_sources": [args.exclude_source],
        "selection_strategy": "retain_all_non_excluded_episodes_then_add_diverse_replacements",
        "base_manifest_sha256": hashlib.sha256(
            (args.base_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "proposal_manifest_sha256": hashlib.sha256(
            (args.proposal_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "episodes": final_records,
    }
    write_json_atomic(args.output_root / "manifest.json", manifest)
    report = {
        "schema_version": proposal_report["schema_version"],
        "episode_count": len(final_records),
        "source_dataset_count": len(final_counts),
        "source_counts": dict(sorted(final_counts.items())),
        "unique_normalized_goals": len(goal_counts),
        "maximum_exact_goal_repetitions": max(goal_counts.values()),
        "exact_visual_signatures": len(used_signatures),
        "source_quotas": quotas,
        "excluded_sources": [args.exclude_source],
        "retained_episode_count": len(retained),
        "replacement_episode_count": len(additions),
        "replacement_source_counts": dict(
            sorted(Counter(record["dataset_name"] for record in additions).items())
        ),
    }
    write_json_atomic(args.output_root / "selection_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
