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


def _box_match(mask: np.ndarray, box: list[int], coordinate_scale: int) -> tuple[int, float, float]:
    height, width = mask.shape
    x_min = max(0, min(width, int(np.floor(box[0] * width / coordinate_scale))))
    y_min = max(0, min(height, int(np.floor(box[1] * height / coordinate_scale))))
    x_max = max(0, min(width, int(np.ceil(box[2] * width / coordinate_scale))))
    y_max = max(0, min(height, int(np.ceil(box[3] * height / coordinate_scale))))
    box_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    mask_area = int(mask.sum())
    if box_area == 0 or mask_area == 0:
        return 0, 0.0, float("inf")
    intersection = int(mask[y_min:y_max, x_min:x_max].sum())
    overlap = 0.0
    if intersection:
        dice = 2.0 * intersection / (mask_area + box_area)
        containment = intersection / mask_area
        overlap = dice + containment
    ys, xs = np.nonzero(mask)
    mask_center_x = (float(xs.min()) + float(xs.max())) / 2.0
    mask_center_y = (float(ys.min()) + float(ys.max())) / 2.0
    box_center_x = (x_min + x_max) / 2.0
    box_center_y = (y_min + y_max) / 2.0
    center_distance = np.hypot(
        (mask_center_x - box_center_x) / max(1, width),
        (mask_center_y - box_center_y) / max(1, height),
    )
    return intersection, overlap, float(center_distance)


def _run_qwen_verified_text_prompts(
    model: Any,
    frames: Path,
    prompts: list[str],
    frame_grounding: list[dict[str, Any]],
    frame_count: int,
    coordinate_scale: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    grounding = {
        int(row["frame_index"]): row["bbox_xyxy_1000"]
        for row in frame_grounding
    }
    if set(grounding) != set(range(frame_count)):
        raise ValueError("Qwen frame grounding does not match the video frames")
    unique_prompts = list(dict.fromkeys(prompt.strip() for prompt in prompts if prompt.strip()))
    candidates = []
    for prompt_index, prompt in enumerate(unique_prompts):
        state = model.init_state(
            resource_path=str(frames),
            offload_video_to_cpu=True,
            async_loading_frames=False,
            video_loader_type="cv2",
        )
        model.add_prompt(state, frame_idx=0, text_str=prompt)
        prompt_candidates: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        prompt_ranks: dict[int, list[float]] = defaultdict(list)
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
                    prompt_candidates[int(object_id)][int(frame_index)] = mask
                    prompt_ranks[int(object_id)].append(float(native_ranks[index]))
        del state
        for object_id, masks in prompt_candidates.items():
            candidates.append({
                "prompt": prompt,
                "prompt_index": prompt_index,
                "object_id": object_id,
                "masks": masks,
                "native_rank": sum(prompt_ranks[object_id]) / max(1, len(prompt_ranks[object_id])),
            })
    if not candidates:
        return {}, {
            "status": "not_found",
            "selection_method": "qwen_all_frame_boxes_over_sam3_text_candidates",
            "candidate_track_count": 0,
            "candidate_prompts": unique_prompts,
            "grounded_frame_count": sum(box is not None for box in grounding.values()),
        }

    def global_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        matched = 0
        overlap_sum = 0.0
        center_distance_sum = 0.0
        compared = 0
        for frame_index, box in grounding.items():
            mask = candidate["masks"].get(frame_index)
            if box is None or mask is None:
                continue
            intersection, overlap, center_distance = _box_match(mask, box, coordinate_scale)
            matched += int(intersection > 0)
            overlap_sum += overlap
            center_distance_sum += center_distance
            compared += 1
        return (
            -matched,
            -overlap_sum,
            center_distance_sum / max(1, compared),
            -len(candidate["masks"]),
            -candidate["native_rank"],
            candidate["prompt_index"],
            candidate["object_id"],
        )

    ordered = sorted(candidates, key=global_key)
    candidate_order = {id(candidate): index for index, candidate in enumerate(ordered)}
    global_candidate = ordered[0]
    masks: dict[int, np.ndarray] = {}
    selected_sources = []
    disjoint = 0
    for frame_index in range(frame_count):
        box = grounding[frame_index]
        available = [candidate for candidate in candidates if frame_index in candidate["masks"]]
        selected = None
        if box is not None and available:
            grounded = []
            for candidate in available:
                intersection, overlap, center_distance = _box_match(
                    candidate["masks"][frame_index], box, coordinate_scale,
                )
                grounded.append((
                    -int(intersection > 0),
                    -overlap,
                    center_distance,
                    candidate_order[id(candidate)],
                    candidate["prompt_index"],
                    candidate["object_id"],
                    candidate,
                ))
            selected_row = min(grounded)
            selected = selected_row[-1]
            disjoint += int(selected_row[0] == 0)
        elif box is None and frame_index in global_candidate["masks"]:
            selected = global_candidate
        if selected is not None:
            masks[frame_index] = selected["masks"][frame_index]
            selected_sources.append((selected["prompt"], selected["object_id"]))

    global_source = (global_candidate["prompt"], global_candidate["object_id"])
    return masks, {
        "status": "tracked" if masks else "not_found",
        "selection_method": "qwen_all_frame_boxes_over_sam3_text_candidates",
        "selected_native_track_id": global_candidate["object_id"],
        "selected_prompt": global_candidate["prompt"],
        "candidate_track_count": len(candidates),
        "candidate_prompts": unique_prompts,
        "visible_frame_count": len(masks),
        "grounded_frame_count": sum(box is not None for box in grounding.values()),
        "qwen_disjoint_frame_count": disjoint,
        "alternate_candidate_frame_count": sum(source != global_source for source in selected_sources),
    }


def _run_box_prompt(
    model: Any,
    frames: Path,
    grounding: dict[str, Any],
    frame_count: int,
    coordinate_scale: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    frame_index = int(grounding["frame_index"])
    if not 0 <= frame_index < frame_count:
        raise ValueError(f"Grounding frame {frame_index} is outside a {frame_count}-frame video")
    x_min, y_min, x_max, y_max = grounding["bbox_xyxy_1000"]
    box_xywh = np.asarray([[
        x_min / coordinate_scale,
        y_min / coordinate_scale,
        (x_max - x_min) / coordinate_scale,
        (y_max - y_min) / coordinate_scale,
    ]], dtype=np.float32)
    candidates: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    ranks: dict[int, list[float]] = defaultdict(list)

    # Use separate inference states for each direction. This prevents the forward
    # action history from turning the reverse pass into a cache-only fetch.
    directions = [(False, frame_count - frame_index)]
    if frame_index > 0:
        directions.append((True, frame_index))
    for reverse, length in directions:
        state = model.init_state(
            resource_path=str(frames),
            offload_video_to_cpu=True,
            async_loading_frames=False,
            video_loader_type="cv2",
        )
        model.add_prompt(
            state,
            frame_idx=frame_index,
            boxes_xywh=box_xywh,
            box_labels=np.asarray([1], dtype=np.int64),
        )
        for output_frame_index, model_output in model.propagate_in_video(
            state,
            start_frame_idx=frame_index,
            max_frame_num_to_track=length,
            reverse=reverse,
        ):
            object_ids = _as_numpy(model_output["out_obj_ids"]).reshape(-1)
            masks = _as_numpy(model_output["out_binary_masks"])
            native_ranks = _as_numpy(model_output["out_probs"]).reshape(-1)
            for candidate_index, object_id in enumerate(object_ids):
                mask = np.asarray(masks[candidate_index]).squeeze().astype(bool)
                if mask.any():
                    candidates[int(object_id)][int(output_frame_index)] = mask
                    ranks[int(object_id)].append(float(native_ranks[candidate_index]))
        del state
    if not candidates:
        return {}, {
            "status": "not_found",
            "selected_native_track_id": None,
            "candidate_track_count": 0,
            "grounding_frame_index": frame_index,
            "grounding_bbox_xyxy_1000": grounding["bbox_xyxy_1000"],
        }
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
        "grounding_frame_index": frame_index,
        "grounding_bbox_xyxy_1000": grounding["bbox_xyxy_1000"],
    }


def _prepare_sam2_jpegs(frame_paths: list[Path], target: Path) -> None:
    target.mkdir()
    for frame_index, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as image:
            image.convert("RGB").save(
                target / f"{frame_index:05d}.jpg",
                format="JPEG",
                quality=100,
                subsampling=0,
            )


def _run_sam2_box_prompt(
    model: Any,
    jpeg_frames: Path,
    grounding: dict[str, Any],
    frame_count: int,
    width: int,
    height: int,
    coordinate_scale: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    frame_index = int(grounding["frame_index"])
    if not 0 <= frame_index < frame_count:
        raise ValueError(f"Grounding frame {frame_index} is outside a {frame_count}-frame video")
    x_min, y_min, x_max, y_max = grounding["bbox_xyxy_1000"]
    box = np.asarray(
        [
            x_min * width / coordinate_scale,
            y_min * height / coordinate_scale,
            x_max * width / coordinate_scale,
            y_max * height / coordinate_scale,
        ],
        dtype=np.float32,
    )
    masks: dict[int, np.ndarray] = {}

    # As in the SAM3 box experiment, each temporal direction starts from a
    # fresh state. The only changed variable in this experiment is the model.
    directions = [(False, frame_count - frame_index)]
    if frame_index > 0:
        directions.append((True, frame_index))
    for reverse, length in directions:
        state = model.init_state(
            video_path=str(jpeg_frames),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        model.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_index,
            obj_id=1,
            box=box,
        )
        for output_frame_index, object_ids, mask_logits in model.propagate_in_video(
            state,
            start_frame_idx=frame_index,
            max_frame_num_to_track=length,
            reverse=reverse,
        ):
            ids = list(object_ids)
            if 1 not in ids:
                continue
            mask_index = ids.index(1)
            mask = _as_numpy(mask_logits[mask_index] > 0.0).squeeze().astype(bool)
            if mask.any():
                masks[int(output_frame_index)] = mask
        del state
    return masks, {
        "status": "tracked" if masks else "not_found",
        "selected_native_track_id": 1 if masks else None,
        "candidate_track_count": 1 if masks else 0,
        "visible_frame_count": len(masks),
        "grounding_frame_index": frame_index,
        "grounding_bbox_xyxy_1000": grounding["bbox_xyxy_1000"],
    }


def _qwen_box_chunks(
    frame_grounding: list[dict[str, Any]],
    frame_count: int,
    chunk_size: int,
) -> list[tuple[int, int, list[dict[str, Any]]]]:
    if chunk_size < 1:
        raise ValueError("SAM2 tracking chunk size must be positive")
    grounding = {int(row["frame_index"]): row for row in frame_grounding}
    if set(grounding) != set(range(frame_count)):
        raise ValueError("Qwen frame grounding does not match the video frames")
    return [
        (
            start,
            min(frame_count, start + chunk_size),
            [
                grounding[index]
                for index in range(start, min(frame_count, start + chunk_size))
                if grounding[index]["bbox_xyxy_1000"] is not None
            ],
        )
        for start in range(0, frame_count, chunk_size)
    ]


def _sam2_box(
    grounding: dict[str, Any],
    width: int,
    height: int,
    coordinate_scale: int,
) -> np.ndarray:
    x_min, y_min, x_max, y_max = grounding["bbox_xyxy_1000"]
    return np.asarray(
        [
            x_min * width / coordinate_scale,
            y_min * height / coordinate_scale,
            x_max * width / coordinate_scale,
            y_max * height / coordinate_scale,
        ],
        dtype=np.float32,
    )


def _run_sam2_qwen_box_chunks(
    model: Any,
    jpeg_frames: Path,
    frame_grounding: list[dict[str, Any]],
    frame_count: int,
    width: int,
    height: int,
    coordinate_scale: int,
    chunk_size: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    masks: dict[int, np.ndarray] = {}
    chunks = _qwen_box_chunks(frame_grounding, frame_count, chunk_size)
    grounded_frame_count = sum(len(rows) for _, _, rows in chunks)
    processed_chunk_count = 0

    def run_direction(
        rows: list[dict[str, Any]],
        start_frame: int,
        stop_frame: int,
        reverse: bool,
    ) -> dict[int, np.ndarray]:
        state = model.init_state(
            video_path=str(jpeg_frames),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        try:
            for row in rows:
                model.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(row["frame_index"]),
                    obj_id=1,
                    box=_sam2_box(row, width, height, coordinate_scale),
                )
            result: dict[int, np.ndarray] = {}
            for output_frame_index, object_ids, mask_logits in model.propagate_in_video(
                state,
                start_frame_idx=start_frame,
                max_frame_num_to_track=abs(stop_frame - start_frame),
                reverse=reverse,
            ):
                ids = list(object_ids)
                if 1 not in ids:
                    continue
                mask_index = ids.index(1)
                mask = _as_numpy(mask_logits[mask_index] > 0.0).squeeze().astype(bool)
                if mask.any():
                    result[int(output_frame_index)] = mask
            return result
        finally:
            del state

    for chunk_start, chunk_end, rows in chunks:
        if not rows:
            continue
        processed_chunk_count += 1
        first_anchor = int(rows[0]["frame_index"])
        last_anchor = int(rows[-1]["frame_index"])
        # Every Qwen box is a conditioning prompt. Forward propagation fills
        # gaps after the first box; a fresh reverse state fills any leading gap.
        forward = run_direction(rows, first_anchor, chunk_end - 1, False)
        if first_anchor > chunk_start:
            masks.update(run_direction(rows, last_anchor, chunk_start, True))
        masks.update(forward)

    return masks, {
        "status": "tracked" if masks else "not_found",
        "selected_native_track_id": 1 if masks else None,
        "candidate_track_count": 1 if masks else 0,
        "visible_frame_count": len(masks),
        "grounded_frame_count": grounded_frame_count,
        "chunk_size": chunk_size,
        "processed_chunk_count": processed_chunk_count,
        "selection_method": "qwen_boxes_as_sam2_conditioning_frames",
    }


def _prefer_primary_masks(
    primary: dict[int, np.ndarray],
    fallback: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], list[int]]:
    fallback_frames = sorted(set(fallback) - set(primary))
    merged = dict(fallback)
    merged.update(primary)
    return merged, fallback_frames


def _build_sam2(config: dict[str, Any]) -> Any:
    model_config = config["sam2"]
    checkpoint = Path(model_config["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_hash = sha256_path(checkpoint)
    if actual_hash != model_config["checkpoint_sha256"]:
        raise ValueError(f"SAM2 checkpoint hash mismatch: {actual_hash}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor(
        model_config["model_config"],
        str(checkpoint),
        device="cuda:0",
        mode="eval",
        apply_postprocessing=True,
        vos_optimized=False,
    )


def _build_sam3(config: dict[str, Any]) -> Any:
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


def build_model(config: dict[str, Any]) -> Any:
    backend = config.get("segmenter", {}).get("backend", "sam3")
    if backend == "sam2":
        return _build_sam2(config)
    if backend == "sam3":
        return _build_sam3(config)
    if backend == "sam3_sam2":
        return {"sam3": _build_sam3(config), "sam2": _build_sam2(config)}
    raise ValueError(f"Unknown segmenter backend: {backend}")


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
        raise ValueError("Prepared frame count changed before segmentation stage")
    import torch
    if model is None:
        model = build_model(config)
    temporary_root = Path(tempfile.mkdtemp(prefix=".masks.", dir=output))
    temporary_root.chmod(0o755)
    height, width = int(context["height"]), int(context["width"])
    track_records = []
    backend = config.get("segmenter", {}).get("backend", "sam3")
    model_config = config[backend]
    prompt_mode = model_config.get("prompt_mode", "text")
    sam2_jpegs = temporary_root / "_sam2_frames" if backend in {"sam2", "sam3_sam2"} else None
    if sam2_jpegs is not None:
        _prepare_sam2_jpegs(frame_paths, sam2_jpegs)
    try:
        for entity in task["entities"]:
            entity_dir = temporary_root / entity["entity_id"]
            entity_dir.mkdir()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if backend == "sam2" and prompt_mode == "qwen_bbox":
                        masks, track = _run_sam2_box_prompt(
                            model,
                            sam2_jpegs,
                            entity["grounding"],
                            len(frame_paths),
                            width,
                            height,
                            int(model_config.get("box_coordinate_scale", 1000)),
                        )
                    elif backend == "sam2" and prompt_mode == "qwen_frame_boxes_chunked":
                        masks, track = _run_sam2_qwen_box_chunks(
                            model,
                            sam2_jpegs,
                            entity["frame_grounding"],
                            len(frame_paths),
                            width,
                            height,
                            int(model_config.get("box_coordinate_scale", 1000)),
                            int(model_config.get("tracking_chunk_size", 5)),
                        )
                    elif backend == "sam3" and prompt_mode == "qwen_bbox":
                        masks, track = _run_box_prompt(
                            model,
                            frames,
                            entity["grounding"],
                            len(frame_paths),
                            int(model_config.get("box_coordinate_scale", 1000)),
                        )
                    elif backend == "sam3" and prompt_mode == "text":
                        masks, track = _run_prompt(model, frames, entity["sam_prompt"], len(frame_paths))
                    elif backend == "sam3" and prompt_mode == "qwen_track_verified_text":
                        masks, track = _run_qwen_verified_text_prompts(
                            model,
                            frames,
                            [
                                entity[field]
                                for field in model_config.get(
                                    "candidate_prompts", ["sam_prompt", "canonical_name"],
                                )
                            ],
                            entity["frame_grounding"],
                            len(frame_paths),
                            int(model_config.get("box_coordinate_scale", 1000)),
                        )
                    elif backend == "sam3_sam2" and prompt_mode == "qwen_verified_text_with_chunked_box_fallback":
                        sam3_masks, sam3_track = _run_qwen_verified_text_prompts(
                            model["sam3"],
                            frames,
                            [
                                entity[field]
                                for field in config["sam3"].get(
                                    "candidate_prompts", ["sam_prompt", "canonical_name"],
                                )
                            ],
                            entity["frame_grounding"],
                            len(frame_paths),
                            int(config["sam3"].get("box_coordinate_scale", 1000)),
                        )
                        sam2_masks, sam2_track = _run_sam2_qwen_box_chunks(
                            model["sam2"],
                            sam2_jpegs,
                            entity["frame_grounding"],
                            len(frame_paths),
                            width,
                            height,
                            int(config["sam2"].get("box_coordinate_scale", 1000)),
                            int(config["sam2"].get("tracking_chunk_size", 5)),
                        )
                        masks, fallback_frames = _prefer_primary_masks(sam3_masks, sam2_masks)
                        track = {
                            "status": "tracked" if masks else "not_found",
                            "visible_frame_count": len(masks),
                            "sam3_visible_frame_count": len(sam3_masks),
                            "sam2_visible_frame_count": len(sam2_masks),
                            "fallback_frame_count": len(fallback_frames),
                            "fallback_frame_indices": fallback_frames,
                            "sam3_candidate_track_count": sam3_track.get("candidate_track_count", 0),
                            "sam2_grounded_frame_count": sam2_track.get("grounded_frame_count", 0),
                            "selection_method": "qwen_verified_sam3_then_sam2_only_when_sam3_missing",
                        }
                    else:
                        raise ValueError(f"Unsupported {backend} prompt mode: {prompt_mode}")
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
                "prompt_type": (
                    "box" if prompt_mode == "qwen_bbox"
                    else "all_frame_boxes"
                    if prompt_mode == "qwen_frame_boxes_chunked"
                    else "text_with_box_fallback"
                    if prompt_mode == "qwen_verified_text_with_chunked_box_fallback"
                    else "text_with_qwen_track_verifier"
                    if prompt_mode == "qwen_track_verified_text"
                    else "text"
                ),
                **track,
                "frames": per_frame,
            })
            gc.collect()
            torch.cuda.empty_cache()
        if sam2_jpegs is not None:
            shutil.rmtree(sam2_jpegs)
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
                "model": (
                    "SAM3 + SAM2.1 fallback"
                    if backend == "sam3_sam2"
                    else "SAM2.1"
                    if backend == "sam2"
                    else "SAM3"
                ),
                "backend": backend,
                "source_revision": model_config["source_revision"],
                "checkpoint_sha256": model_config["checkpoint_sha256"],
                "selection": (
                    "Qwen boxes as SAM2 conditioning frames in independent temporal chunks"
                    if backend == "sam2" and prompt_mode == "qwen_frame_boxes_chunked"
                    else "Qwen-verified SAM3 primary; chunked Qwen-box SAM2 only for missing frames"
                    if backend == "sam3_sam2"
                    else "single Qwen-grounded object track"
                    if backend == "sam2"
                    else "maximum temporal persistence, then mean native ranking, then lower native track id"
                ),
                "prompt_mode": prompt_mode,
                "box_propagation": model_config.get("box_propagation") if prompt_mode in {"qwen_bbox", "qwen_frame_boxes_chunked", "qwen_verified_text_with_chunked_box_fallback"} else None,
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
