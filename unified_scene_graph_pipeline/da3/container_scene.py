#!/usr/bin/env python3
"""Run the pinned DA3 streaming implementation atomically for one prepared scene."""

from __future__ import annotations

import argparse
import gc
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
from typing import Any

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


class NativeRunner:
    """Own one DA3 model and reuse it for every scene in this process."""

    def __init__(self, config_path: Path) -> None:
        streaming_root = Path("/opt/da3/da3_streaming")
        sys.path.insert(0, str(streaming_root))
        from da3_streaming import (  # type: ignore[import-not-found]
            DA3_Streaming,
            load_config,
            load_da3_model,
            merge_ply_files,
            warmup_numba,
        )

        self._streaming_class = DA3_Streaming
        self._load_model = load_da3_model
        self._merge_ply_files = merge_ply_files
        self._config = load_config(str(config_path))
        self._model = None
        if self._config["Model"]["align_lib"] == "numba":
            warmup_numba()

    def run(self, image_dir: Path, output_dir: Path) -> None:
        import torch

        if self._model is None:
            self._model = self._load_model(self._config)
        worker = self._streaming_class(
            str(image_dir), str(output_dir), self._config, model=self._model
        )
        try:
            worker.run()
            worker.close()
            print("Saving all the point clouds")
            self._merge_ply_files(
                str(output_dir / "pcd"), str(output_dir / "pcd" / "combined_pcd.ply")
            )
            print("DA3-Streaming done.")
        finally:
            del worker
            torch.cuda.empty_cache()
            gc.collect()


def process_scene(
    scene: Path, config_path: Path, overwrite: bool, runner: NativeRunner
) -> dict[str, Any]:
    scene = scene.resolve()
    frames = natural_frames(scene / "frames")
    if not frames:
        raise ValueError(f"No prepared frames: {scene}")
    final = scene / "da3"
    backups = sorted(scene.glob(".da3.backup.*"))
    if not final.exists() and backups:
        backups[-1].replace(final)
    stage_fingerprint = fingerprint(frames, config_path)
    marker = scene / ".stages" / "da3.json"
    if final.exists() and marker.is_file() and not overwrite:
        marker_value = json.loads(marker.read_text())
        if marker_value.get("fingerprint") == stage_fingerprint:
            details = validate(final, len(frames))
            result = {"status": "skipped_valid", **details}
            print("DA3_RESULT=" + json.dumps(result, sort_keys=True))
            return result
    # Every Docker invocation starts this process as PID 1. A PID-based name
    # therefore collides with a partial directory left by an interrupted run.
    temporary = Path(tempfile.mkdtemp(prefix=".da3.tmp.", dir=scene))
    started = time.monotonic()
    try:
        runner.run(scene / "frames", temporary)
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
            "max_rss_kb": max(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            ),
            **details,
        }
        write_json(marker, {"stage": "da3", "fingerprint": stage_fingerprint, **result})
        update_report(scene, "success", result)
        print("DA3_RESULT=" + json.dumps(result, sort_keys=True))
        return result
    except Exception:
        failed = scene / f".da3.failed.{int(time.time())}"
        if temporary.exists():
            temporary.replace(failed)
        update_report(scene, "failed", {"error": "DA3 stage failed; see container log"})
        raise


def manifest_scenes(manifest: Path, output_root: Path) -> list[Path]:
    value = json.loads(manifest.read_text())
    scenes = []
    for item in value.get("episodes", value.get("scenes", [])):
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative_path in manifest: {relative}")
        scenes.append(output_root / relative)
    return scenes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--config", default="/pipeline/da3/config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    scenes = [Path(value) for value in args.scene]
    if args.manifest:
        if not args.output_root:
            parser.error("--output-root is required with --manifest")
        scenes.extend(manifest_scenes(Path(args.manifest), Path(args.output_root)))
    if not scenes:
        parser.error("provide --scene or --manifest")

    runner = NativeRunner(Path(args.config))
    failures = []
    completed = 0
    skipped = 0
    for index, scene in enumerate(scenes, start=1):
        print(f"DA3_BEGIN index={index}/{len(scenes)} scene={scene}")
        try:
            result = process_scene(
                scene, Path(args.config), args.overwrite, runner
            )
            skipped += result["status"] == "skipped_valid"
            completed += result["status"] == "success"
        except Exception as error:
            failures.append({"scene": str(scene), "error": str(error)})
            print("DA3_ERROR=" + json.dumps(failures[-1], sort_keys=True), file=sys.stderr)
            if args.fail_fast:
                raise
    summary = {
        "scene_count": len(scenes),
        "success_count": completed,
        "skipped_count": skipped,
        "failure_count": len(failures),
        "failures": failures,
    }
    print("DA3_BATCH_RESULT=" + json.dumps(summary, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
