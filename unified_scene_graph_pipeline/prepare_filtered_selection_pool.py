#!/usr/bin/env python3
"""Derive a reproducible candidate pool after excluding one source dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def allocate_quotas(
    counts: dict[str, int], total: int, minimum: int, maximum: int,
) -> dict[str, int]:
    capacity = sum(min(count, maximum) for count in counts.values())
    if total > capacity:
        raise ValueError(f"Requested {total} scenes, but capped capacity is {capacity}")
    quotas = {name: min(count, minimum, maximum) for name, count in counts.items()}
    if sum(quotas.values()) > total:
        raise ValueError("Minimum quotas exceed the requested total")
    while sum(quotas.values()) < total:
        remaining = total - sum(quotas.values())
        eligible = [
            name for name, count in counts.items()
            if quotas[name] < min(count, maximum)
        ]
        weights = {name: math.sqrt(counts[name]) for name in eligible}
        weight_sum = sum(weights.values())
        raw = {name: remaining * weights[name] / weight_sum for name in eligible}
        increments = {
            name: min(
                int(math.floor(raw[name])),
                min(counts[name], maximum) - quotas[name],
            )
            for name in eligible
        }
        if sum(increments.values()) == 0:
            name = max(eligible, key=lambda key: (raw[key], counts[key], key))
            increments[name] = 1
        for name, increment in increments.items():
            applied = min(increment, remaining)
            quotas[name] += applied
            remaining -= applied
            if remaining == 0:
                break
    return dict(sorted(quotas.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-work", type=Path, required=True)
    parser.add_argument("--output-work", type=Path, required=True)
    parser.add_argument("--exclude-source", required=True)
    parser.add_argument("--final-size", type=int, default=1000)
    parser.add_argument("--minimum-per-source", type=int, default=40)
    parser.add_argument("--maximum-per-source", type=int, default=151)
    args = parser.parse_args()

    args.output_work.mkdir(parents=True, exist_ok=True)
    original_candidates = read_jsonl(args.source_work / "candidate_manifest.jsonl")
    candidates = [
        record for record in original_candidates
        if record["dataset_name"] != args.exclude_source
    ]
    if len(candidates) == len(original_candidates):
        raise ValueError(f"Excluded source is absent: {args.exclude_source}")

    registry = read_json(args.source_work / "registry_summary.json")
    source_counts = {
        name: int(count)
        for name, count in registry["source_counts"].items()
        if name != args.exclude_source
        and any(record["dataset_name"] == name for record in candidates)
    }
    quotas = allocate_quotas(
        source_counts,
        args.final_size,
        args.minimum_per_source,
        args.maximum_per_source,
    )
    candidate_counts = Counter(record["dataset_name"] for record in candidates)
    insufficient = {
        name: {"quota": quota, "candidates": candidate_counts[name]}
        for name, quota in quotas.items()
        if candidate_counts[name] < quota
    }
    if insufficient:
        raise ValueError(f"Existing candidate pool is insufficient: {insufficient}")

    visual = np.load(args.source_work / "visual_embeddings.npz")
    original_ids = [str(value) for value in visual["episode_ids"]]
    expected_ids = [record["episode_id"] for record in original_candidates]
    if original_ids != expected_ids:
        raise ValueError("Visual embedding order does not match the candidate manifest")
    keep = np.asarray(
        [record["dataset_name"] != args.exclude_source for record in original_candidates],
        dtype=bool,
    )
    np.savez_compressed(
        args.output_work / "visual_embeddings.npz",
        episode_ids=visual["episode_ids"][keep],
        embeddings=visual["embeddings"][keep],
        signatures=visual["signatures"][keep],
    )
    write_jsonl_atomic(args.output_work / "candidate_manifest.jsonl", candidates)

    original_shortlist = read_json(args.source_work / "shortlist_report.json")
    report = {
        "schema_version": original_shortlist["schema_version"],
        "seed": original_shortlist["seed"],
        "text_model": original_shortlist["text_model"],
        "final_size": args.final_size,
        "candidate_multiplier": original_shortlist["candidate_multiplier"],
        "included_sources": sorted(quotas),
        "excluded_sources": [args.exclude_source],
        "candidate_count": len(candidates),
        "final_source_quotas": quotas,
        "candidate_source_counts": dict(sorted(candidate_counts.items())),
        "candidate_unique_goals": len({record["normalized_goal"] for record in candidates}),
        "pool_derivation": "filtered_existing_text_and_visual_diversity_candidates",
        "source_candidate_manifest_sha256": hashlib.sha256(
            (args.source_work / "candidate_manifest.jsonl").read_bytes()
        ).hexdigest(),
    }
    write_json_atomic(args.output_work / "shortlist_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
