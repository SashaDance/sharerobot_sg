#!/usr/bin/env python3
"""Run the pinned DA3 streaming implementation atomically for one prepared scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def natural_frames(directory: Path) -> list[Path]:
    return sorted(
        [*directory.glob("*.png"), *directory.glob("*.jpg"), *directory.glob("*.jpeg")],
        key=lambda path: (int(path.stem.rsplit("_", 1)[-1]), path.name),
    )


def validate(output: Path, expected: int) -> dict:
    npzs = sorted((output / "results_output").glob("frame_*.npz"), key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
    if len(npzs) != expected:
        raise RuntimeError(f"expected {expected} NPZ files, found {len(npzs)}")
    for path in npzs:
        with np.load(path) as item:
            for name in ("image", "depth", "intrinsics"):
                if name not in item:
                    raise RuntimeError(f"{path}: missing {name}")
                if name != "image" and not np.isfinite(item[name]).all():
                    raise RuntimeError(f"{path}: non-finite {name}")
    pose_lines = [line for line in (output / "camera_poses.txt").read_text().splitlines() if line.strip()]
    pose_values = [np.fromstring(line, sep=" ") for line in pose_lines]
    if len(pose_lines) != expected or any(values.size != 16 or not np.isfinite(values).all() for values in pose_values):
        raise RuntimeError(f"camera_poses.txt: expected {expected} lines, found {len(pose_lines)}")
    for name in ("camera_poses.ply", "pcd/combined_pcd.ply"):
        path = output / name
        if not path.is_file() or path.stat().st_size < 200:
            raise RuntimeError(f"{name}: missing or empty")
    return {
        "npz_count": len(npzs),
        "pose_count": len(pose_lines),
        "pointcloud_fallback": (output / "pcd" / "fallback.json").is_file(),
    }


def repair_empty_pointcloud(output: Path) -> bool:
    """Regenerate a degenerate native point cloud from valid DA3 frame outputs."""
    combined = output / "pcd" / "combined_pcd.ply"
    if combined.is_file() and combined.stat().st_size >= 200:
        return False
    subprocess.run([
        sys.executable,
        "/opt/da3/da3_streaming/npz_output_process.py",
        "--npz_folder", str(output / "results_output"),
        "--pose_file", str(output / "camera_poses.txt"),
        "--output_file", str(combined),
        "--conf_threshold_coef", "0.75",
        # Very low-confidence clips may contain too few valid points for the
        # native sampling ratio. Retain all valid points instead of rounding
        # the requested sample count to zero.
        "--sample_ratio", "1.0",
    ], check=True)
    if not combined.is_file() or combined.stat().st_size < 200:
        raise RuntimeError("point-cloud fallback produced no valid vertices")
    shutil.copy2(combined, output / "pcd" / "0_pcd.ply")
    write_json(output / "pcd" / "fallback.json", {
        "reason": "native confidence-filtered point cloud was empty",
        "method": "DA3 npz_output_process with full valid-point retention",
    })
    return True


def fingerprint(frames: list[Path], config: Path) -> str:
    digest = hashlib.sha256(b"da3_stage_v2_empty_pointcloud_fallback")
    for path in [config, *frames]:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            while block := stream.read(8 << 20):
                digest.update(block)
    digest.update(b"3d835ec1a5802d64a8b8b15f817a1ab54809bfe4")
    digest.update(b"8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c")
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def update_report(scene: Path, status: str, details: dict) -> None:
    path = scene / "run_report.json"
    report = json.loads(path.read_text()) if path.is_file() else {"schema_version": "unified_sgg_run_report_v1", "stages": {}}
    report.setdefault("stages", {})["da3"] = {"status": status, **details}
    write_json(path, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", default="/pipeline/da3/config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    scene = Path(args.scene).resolve()
    frames = natural_frames(scene / "frames")
    if not frames:
        raise SystemExit("No prepared frames")
    final = scene / "da3"
    backups = sorted(scene.glob(".da3.backup.*"))
    if not final.exists() and backups:
        backups[-1].replace(final)
    config_path = Path(args.config)
    stage_fingerprint = fingerprint(frames, config_path)
    marker = scene / ".stages" / "da3.json"
    if final.exists() and marker.is_file() and not args.overwrite:
        marker_value = json.loads(marker.read_text())
        if marker_value.get("fingerprint") == stage_fingerprint:
            details = validate(final, len(frames))
            print("DA3_RESULT=" + json.dumps({"status": "skipped_valid", **details}, sort_keys=True))
            return 0
    # Every Docker invocation starts this process as PID 1. A PID-based name
    # therefore collides with a partial directory left by an interrupted run.
    temporary = Path(tempfile.mkdtemp(prefix=".da3.tmp.", dir=scene))
    started = time.monotonic()
    try:
        subprocess.run([
            sys.executable, "/opt/da3/da3_streaming/da3_streaming.py",
            "--image_dir", str(scene / "frames"), "--config", args.config, "--output_dir", str(temporary),
        ], check=True)
        shutil.copy2(config_path, temporary / "config.yaml")
        repair_empty_pointcloud(temporary)
        details = validate(temporary, len(frames))
        if final.exists():
            backup = scene / f".da3.backup.{int(time.time())}"
            final.replace(backup)
            temporary.replace(final)
            shutil.rmtree(backup)
        else:
            temporary.replace(final)
        result = {
            "status": "success", "runtime_seconds": round(time.monotonic() - started, 3),
            "max_rss_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, **details,
        }
        write_json(marker, {"stage": "da3", "fingerprint": stage_fingerprint, **result})
        update_report(scene, "success", result)
        print("DA3_RESULT=" + json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        failed = scene / f".da3.failed.{int(time.time())}"
        if temporary.exists():
            temporary.replace(failed)
        update_report(scene, "failed", {"error": "DA3 stage failed; see container log"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
