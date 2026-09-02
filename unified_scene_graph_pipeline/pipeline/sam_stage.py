from __future__ import annotations

import gc
import os
import shutil
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common import (
    complete_stage,
    config_hash,
    image_paths,
    read_json,
    sha256_path,
    stage_current,
    stage_fingerprint,
    update_run_report,
    write_json_atomic,
)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _run_prompt(model: Any, frames: Path, prompt: str, frame_count: int) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    state = model.init_state(
        resource_path=str(frames),
        offload_video_to_cpu=True,
        async_loading_frames=False,
        video_loader_type="cv2",
    )
    model.add_prompt(state, frame_idx=0, text_str=prompt)
    candidates: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    ranks: dict[int, list[float]] = defaultdict(list)
    for frame_index, output in model.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=frame_count,
        reverse=False,
    ):
        object_ids = _as_numpy(output["out_obj_ids"]).reshape(-1)
        masks = _as_numpy(output["out_binary_masks"])
        native_ranks = _as_numpy(output["out_probs"]).reshape(-1)
        for index, object_id in enumerate(object_ids):
            mask = np.asarray(masks[index]).squeeze().astype(bool)
            if mask.any():
                candidates[int(object_id)][int(frame_index)] = mask
                ranks[int(object_id)].append(float(native_ranks[index]))
    if not candidates:
        return {}, {"status": "not_found", "selected_native_track_id": None, "candidate_track_count": 0}
    # Native scores are used only for deterministic tie-breaking and never leave this process.
    selected = min(
        candidates,
        key=lambda object_id: (
            -len(candidates[object_id]),
            -(sum(ranks[object_id]) / max(1, len(ranks[object_id]))),
            object_id,
        ),
    )
    return candidates[selected], {
        "status": "tracked",
        "selected_native_track_id": selected,
        "candidate_track_count": len(candidates),
        "visible_frame_count": len(candidates[selected]),
    }


def build_model(config: dict[str, Any]) -> Any:
    checkpoint = Path(config["sam3"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_hash = sha256_path(checkpoint)
    if actual_hash != config["sam3"]["checkpoint_sha256"]:
        raise ValueError(f"SAM3 checkpoint hash mismatch: {actual_hash}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from sam3.model_builder import build_sam3_video_model

    bpe = Path(config["sam3"].get("bpe_path", "/opt/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
    return build_sam3_video_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        bpe_path=str(bpe),
        strict_state_dict_loading=True,
        apply_temporal_disambiguation=True,
        device="cuda:0",
        compile=False,
    ).eval()


def segment(output: Path, config: dict[str, Any], overwrite: bool = False, model: Any | None = None) -> dict[str, Any]:
    input_path = output / "input.json"
    task_path = output / "task_spec.json"
    fingerprint = stage_fingerprint(config, [input_path, task_path], "sam")
    target_masks = output / "masks"
    target_tracks = output / "tracks.json"
    backups = sorted(output.glob(".masks.backup.*"))
    if not target_masks.exists() and backups:
        backups[-1].replace(target_masks)
    if not overwrite and stage_current(output, "sam", fingerprint, [target_masks, target_tracks]):
        return read_json(target_tracks)
    context = read_json(input_path)
    task = read_json(task_path)
    frames = output / "frames"
    frame_paths = image_paths(frames)
    if len(frame_paths) != int(context["frame_count"]):
        raise ValueError("Prepared frame count changed before SAM3 stage")
    import torch
    if model is None:
        model = build_model(config)
    temporary_root = Path(tempfile.mkdtemp(prefix=".masks.", dir=output))
    temporary_root.chmod(0o755)
    height, width = int(context["height"]), int(context["width"])
    track_records = []
    try:
        for entity in task["entities"]:
            entity_dir = temporary_root / entity["entity_id"]
            entity_dir.mkdir()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, track = _run_prompt(model, frames, entity["sam_prompt"], len(frame_paths))
            except Exception as error:
                traceback.print_exc()
                masks, track = {}, {
                    "status": "segmentation_failed",
                    "reason": f"{type(error).__name__}: {str(error)[:300]}",
                }
            per_frame = []
            for frame_index in range(len(frame_paths)):
                mask = masks.get(frame_index)
                status = "visible" if mask is not None and mask.any() else (
                    "segmentation_failed" if track["status"] == "segmentation_failed" else "not_visible"
                )
                if mask is None:
                    mask = np.zeros((height, width), dtype=np.uint8)
                elif mask.shape != (height, width):
                    mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)) > 0
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(entity_dir / f"frame_{frame_index:06d}.png")
                per_frame.append({"frame_index": frame_index, "status": status, "area": int(mask.sum())})
            track_records.append({
                "entity_id": entity["entity_id"],
                "role": entity["role"],
                "sam_prompt": entity["sam_prompt"],
                **track,
                "frames": per_frame,
            })
            gc.collect()
            torch.cuda.empty_cache()
        backup = None
        if target_masks.exists():
            backup = output / f".masks.backup.{os.getpid()}"
            target_masks.replace(backup)
        temporary_root.replace(target_masks)
        if backup is not None:
            shutil.rmtree(backup)
        tracks = {
            "schema_version": "unified_sgg_tracks_v1",
            "frame_count": len(frame_paths),
            "tracks": track_records,
            "provenance": {
                "model": "SAM3",
                "source_revision": config["sam3"]["source_revision"],
                "checkpoint_sha256": config["sam3"]["checkpoint_sha256"],
                "selection": "maximum temporal persistence, then mean native ranking, then lower native track id",
                "config_hash": config_hash(config),
            },
        }
        write_json_atomic(target_tracks, tracks)
        complete_stage(output, "sam", fingerprint, {"frame_count": len(frame_paths)})
        update_run_report(output, "sam", "success", {"frame_count": len(frame_paths)})
        return tracks
    except Exception as error:
        shutil.rmtree(temporary_root, ignore_errors=True)
        update_run_report(output, "sam", "failed", {"error": f"{type(error).__name__}: {error}"})
        raise
