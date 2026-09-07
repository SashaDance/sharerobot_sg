from __future__ import annotations

import base64
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests

from common import (
    complete_stage, config_hash, mask_geometry, read_json, require_api_key,
    mask_label_placement, stage_current, stage_fingerprint, update_run_report,
    write_json_atomic,
)


ROLE_NAMES = ("robot", "manipulated_object", "initial_support", "target", "whole_parent")
MASK_COLORS = {
    "robot": ((230, 57, 70), "red"),
    "manipulated_object": ((69, 123, 157), "blue"),
    "initial_support": ((42, 157, 143), "teal"),
    "target": ((244, 162, 97), "orange"),
    "whole_parent": ((138, 43, 226), "purple"),
}
MASK_MARKERS = {
    "robot": "R", "manipulated_object": "M", "initial_support": "I",
    "target": "T", "whole_parent": "P",
}


def contextual_chunks(indices: list[int], maximum_images: int):
    if maximum_images < 2:
        raise ValueError("Qwen frame limit must be at least 2")
    start = 0
    while start < len(indices):
        previous = indices[start - 1] if start else None
        capacity = maximum_images - (1 if previous is not None else 0)
        current = indices[start : start + capacity]
        yield current, previous
        start += len(current)


def sparse_anchor_indices(
    indices: list[int], anchors_per_window: int, window_size: int,
) -> list[int]:
    """Choose deterministic uniform box anchors without dropping video frames."""
    if anchors_per_window < 2:
        raise ValueError("box_anchor_count_per_window must be at least 2")
    if window_size < anchors_per_window:
        raise ValueError("Qwen frame window must be at least the box anchor count")
    selected: list[int] = []
    for start in range(0, len(indices), window_size):
        window = indices[start : start + window_size]
        if len(window) <= anchors_per_window:
            selected.extend(window)
            continue
        positions = [
            math.ceil(slot * (len(window) - 1) / (anchors_per_window - 1))
            for slot in range(anchors_per_window)
        ]
        selected.extend(window[position] for position in dict.fromkeys(positions))
    return selected


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def png_data_url(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def extract_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "")
    decoder = json.JSONDecoder()
    for position, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Qwen response has no JSON object")


class QwenClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config["qwen"]
        self.url = self.config["base_url"].rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {require_api_key()}"}

    def request(
        self,
        content: list[dict[str, Any]],
        max_tokens: int,
        validator: Callable[[dict[str, Any]], Any],
    ) -> tuple[Any, list[dict[str, Any]]]:
        attempts = []
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        for attempt in range(int(self.config["retries"]) + 1):
            started = time.monotonic()
            payload = {
                "model": self.config["model"],
                "messages": messages,
                "temperature": float(self.config["temperature"]),
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
                "response_format": {"type": "json_object"},
            }
            try:
                response = requests.post(self.url, headers=self.headers, json=payload, timeout=600)
            except requests.RequestException as error:
                attempts.append({"attempt": attempt + 1, "seconds": round(time.monotonic() - started, 3), "transport_error": type(error).__name__})
                if attempt >= int(self.config["retries"]):
                    raise
                time.sleep(2 ** attempt)
                continue
            record: dict[str, Any] = {
                "attempt": attempt + 1,
                "status_code": response.status_code,
                "seconds": round(time.monotonic() - started, 3),
            }
            attempts.append(record)
            if not response.ok:
                record["response_excerpt"] = response.text[:500]
                if attempt >= int(self.config["retries"]):
                    response.raise_for_status()
                time.sleep(2 ** attempt)
                continue
            body = response.json()
            record["usage"] = body.get("usage")
            message = body["choices"][0]["message"]
            raw = message.get("content") or ""
            record["raw_response"] = raw
            try:
                return validator(extract_object(raw)), attempts
            except Exception as error:
                record["validation_error"] = str(error)
                # Give the model the actual invalid answer when requesting a repair.
                # Previously, retries mentioned a "prior JSON" without including it,
                # so the model often regenerated the same malformed structure.
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"Your prior JSON was invalid: {error}. "
                            "Correct that exact response and return the complete JSON object only."
                        ),
                    },
                ]
        raise RuntimeError(f"Qwen schema failed after {len(attempts)} attempts: {attempts[-1].get('validation_error')}")


def entity_prompt(
    goal: str, frame_indices: list[int], include_broad_concepts: bool = False,
) -> str:
    entity_shape = (
        '{"canonical_name":"specific visible name","sam_prompt":"specific short noun phrase",'
        '"broad_sam_prompt":"broader common object category"}'
        if include_broad_concepts
        else '{"canonical_name":"specific visible name","sam_prompt":"short visual description"}'
    )
    return f"""You identify task entities in ordered robot-manipulation frames.
Planning goal: {goal}
The following images are frames {frame_indices} in temporal order.
Use visual appearance, not hidden annotations. Return exactly this JSON shape:
{{"roles":{{
 "robot":{entity_shape},
 "manipulated_object":{entity_shape},
 "initial_support":null or {entity_shape},
 "target":null or {entity_shape},
 "whole_parent":null or {entity_shape}
}},"task_actions":["canonical action labels"]}}
Allowed task actions: reach_for, grab, lift, move, place, release, push, pour, open, close, insert, stack.
Robot and manipulated_object must be present. Identify the manipulated object as the physical object whose position or state changes across the ordered frames, using the goal only as context. Initial support is the visible surface or receptacle supporting the manipulated object in the early frames. Target is the visible intended destination surface or receptacle, supported by the goal and the observed motion or final frames. If the goal conflicts with the visible action, trust the video; use null for a target that is neither visible nor supported by the observed action.

Keep visually specific short descriptions: robot sam_prompt must be exactly "robot arm"; other sam_prompt values should normally be a discriminative color plus a common object noun, such as "yellow banana", "brown cup", "red cube", "black clamp", "wooden table", "orange bowl", "silver plate", or "blue container". canonical_name may remain more specific. When broad_sam_prompt is requested, make it a short, visually grounded super-category that SAM3 can detect exhaustively, such as "robot", "banana", "cup", "cube", "clamp", "table", "bowl", "plate", or "container". It must not contain an action or spatial relation. Do not output points, boxes, masks, confidence, planning steps, synonyms, or extra roles."""


def validate_entity_document(
    value: dict[str, Any], include_broad_concepts: bool = False,
) -> dict[str, Any]:
    if set(value) != {"roles", "task_actions"}:
        raise ValueError("Expected only roles and task_actions")
    roles = value["roles"]
    if not isinstance(roles, dict) or set(roles) != set(ROLE_NAMES):
        raise ValueError(f"roles must contain exactly {ROLE_NAMES}")
    for role in ROLE_NAMES:
        entity = roles[role]
        if entity is None:
            if role in {"robot", "manipulated_object"}:
                raise ValueError(f"Required role {role} is null")
            continue
        expected_fields = {"canonical_name", "sam_prompt"}
        if include_broad_concepts:
            expected_fields.add("broad_sam_prompt")
        if not isinstance(entity, dict) or set(entity) != expected_fields:
            raise ValueError(f"Invalid entity shape for {role}")
        if not all(
            isinstance(entity[key], str) and entity[key].strip()
            for key in expected_fields
        ):
            raise ValueError(f"Empty entity value for {role}")
    allowed = {"reach_for", "grab", "lift", "move", "place", "release", "push", "pour", "open", "close", "insert", "stack"}
    actions = value["task_actions"]
    if not isinstance(actions, list) or any(item not in allowed for item in actions):
        raise ValueError("Invalid task action")
    # This role-level normalization is part of the shared pipeline contract. It is
    # deliberately independent of the dataset, episode, and Qwen wording.
    roles["robot"]["sam_prompt"] = "robot arm"
    return value


def tracking_prompt(goal: str, roles: dict[str, Any], frame_indices: list[int]) -> str:
    return f"""You are a visual multi-object tracker and verifier for ordered robot-manipulation frames.
Planning goal: {goal}
Fixed proposed task roles: {json.dumps(roles, ensure_ascii=False)}
The following images are frames {frame_indices} in temporal order.

Return exactly this compact JSON shape and no extra keys:
{{"tracks":{{
 "robot":[[frame_index,[xmin,ymin,xmax,ymax] or null],...],
 "manipulated_object":[[frame_index,[xmin,ymin,xmax,ymax] or null],...],
 "initial_support":null or [[frame_index,[xmin,ymin,xmax,ymax] or null],...],
 "target":null or [[frame_index,[xmin,ymin,xmax,ymax] or null],...],
 "whole_parent":null or [[frame_index,[xmin,ymin,xmax,ymax] or null],...]
}}}}

For every proposed non-null role, return exactly one row for every supplied frame, in order. Track the same physical instance over time. A box must tightly enclose only the visible pixels of that instance and use normalized integer coordinates from [0,0] at top-left to [1000,1000] at bottom-right. Return null only when that entity is completely outside the frame, fully occluded, or cannot be localized without guessing. Do not substitute a same-category distractor, merge multiple instances, include a held object in the robot box, or change the role definitions. For the robot, box all visible robot-arm pixels. Verify manipulated_object identity by its motion across the whole sequence; verify initial_support in early frames and target from the observed destination/final interaction. Do not output masks, confidence, reasoning, actions, relations, planning steps, aliases, or synonyms."""


def validate_tracking_document(
    value: dict[str, Any], requested: list[int], roles: dict[str, Any],
) -> dict[str, list[dict[str, Any]] | None]:
    if set(value) != {"tracks"} or not isinstance(value["tracks"], dict):
        raise ValueError("Expected only tracks")
    tracks = value["tracks"]
    if set(tracks) != set(ROLE_NAMES):
        raise ValueError(f"tracks must contain exactly {ROLE_NAMES}")
    normalized: dict[str, list[dict[str, Any]] | None] = {}
    for role in ROLE_NAMES:
        rows = tracks[role]
        if roles[role] is None:
            if rows is not None:
                raise ValueError(f"Null proposed role {role} must have a null track")
            normalized[role] = None
            continue
        if not isinstance(rows, list) or len(rows) != len(requested):
            raise ValueError(f"Track {role} must contain exactly {len(requested)} rows")
        if any(not isinstance(row, list) or len(row) != 2 for row in rows):
            raise ValueError(f"Each {role} track row must be [frame_index, bbox_or_null]")
        if [row[0] for row in rows] != requested:
            raise ValueError(f"Track {role} must contain requested frame indices in order")
        normalized_rows = []
        for frame_index, box in rows:
            if box is not None and (
                not isinstance(box, list) or len(box) != 4
                or any(not isinstance(coordinate, int) or isinstance(coordinate, bool) for coordinate in box)
                or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)
            ):
                raise ValueError(f"Invalid normalized tracking box for {role} at frame {frame_index}")
            normalized_rows.append({"frame_index": frame_index, "bbox_xyxy_1000": box})
        normalized[role] = normalized_rows
    return normalized


def entity_content(
    goal: str, frames: Path, indices: list[int], include_broad_concepts: bool = False,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": entity_prompt(goal, indices, include_broad_concepts),
    }]
    for index in indices:
        content.extend([
            {"type": "text", "text": f"frame {index}"},
            {"type": "image_url", "image_url": {"url": data_url(frames / f"frame_{index:06d}.png")}},
        ])
    return content


def entity_reflection_prompt(
    goal: str,
    proposal: dict[str, Any],
    frame_indices: list[int],
) -> str:
    legend = {
        role: {"marker": MASK_MARKERS[role], "color": MASK_COLORS[role][1]}
        for role in ROLE_NAMES
    }
    return f"""You are the reflection stage of a zero-shot robot-video grounding system.
Planning goal: {goal}
Initial proposal: {json.dumps(proposal, ensure_ascii=False)}
The following images are all ordered frames {frame_indices}. They show the initial proposal as colored boxes with this legend: {json.dumps(legend)}.

Audit the proposal against the complete visible trajectory, then return a corrected entity proposal. Check that the manipulated object is the single physical instance whose pose or state changes because of the robot; do not select a nearby distractor, tool fixture, repeated row, or object mentioned by the goal but contradicted by the video. Check initial support in early frames and target at the observed destination in late frames. A target may be null when no destination interaction is visible. For repeated instances, use a short visual description that distinguishes the interacted instance. Preserve a role only when its identity is consistent across the sequence.

Return exactly the same JSON shape as the initial proposal:
{{"roles":{{
 "robot":{{"canonical_name":"specific visible name","sam_prompt":"robot arm"}},
 "manipulated_object":{{"canonical_name":"specific visible name","sam_prompt":"short discriminative visual description"}},
 "initial_support":null or {{"canonical_name":"...","sam_prompt":"..."}},
 "target":null or {{"canonical_name":"...","sam_prompt":"..."}},
 "whole_parent":null or {{"canonical_name":"...","sam_prompt":"..."}}
}},"task_actions":["canonical action labels"]}}
Allowed actions: reach_for, grab, lift, move, place, release, push, pour, open, close, insert, stack.
Do not output boxes, points, masks, confidence, reasoning, feedback, planning steps, aliases, or extra keys."""


def tracking_overlay_frame(
    frame_path: Path,
    roles: dict[str, Any],
    frame_grounding: dict[str, list[dict[str, Any]] | None],
    frame_index: int,
    coordinate_scale: int = 1000,
) -> bytes:
    from io import BytesIO
    from PIL import Image, ImageDraw

    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for role in ROLE_NAMES:
        rows = frame_grounding.get(role)
        if roles.get(role) is None or rows is None:
            continue
        row = rows[frame_index]
        box = row.get("bbox_xyxy_1000")
        if box is None:
            continue
        x1 = round(box[0] * width / coordinate_scale)
        y1 = round(box[1] * height / coordinate_scale)
        x2 = round(box[2] * width / coordinate_scale)
        y2 = round(box[3] * height / coordinate_scale)
        color = MASK_COLORS[role][0]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, min(width, height) // 160))
        marker = MASK_MARKERS[role]
        label_box = draw.textbbox((x1, y1), marker)
        draw.rectangle(label_box, fill=(0, 0, 0))
        draw.text((x1, y1), marker, fill=color)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def entity_reflection_content(
    goal: str,
    output: Path,
    proposal: dict[str, Any],
    frame_grounding: dict[str, list[dict[str, Any]] | None],
    indices: list[int],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": entity_reflection_prompt(goal, proposal, indices),
    }]
    for index in indices:
        overlay = tracking_overlay_frame(
            output / "frames" / f"frame_{index:06d}.png",
            proposal["roles"],
            frame_grounding,
            index,
        )
        content.extend([
            {"type": "text", "text": f"frame {index}"},
            {"type": "image_url", "image_url": {"url": png_data_url(overlay)}},
        ])
    return content


def ground_roles_in_chunks(
    client: QwenClient,
    goal: str,
    roles: dict[str, Any],
    output: Path,
    indices: list[int],
    tracking_chunk_size: int,
    max_tokens: int,
    query_indices: list[int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]] | None], list[dict[str, Any]]]:
    frame_grounding: dict[str, list[dict[str, Any]] | None] = {
        role: ([] if roles[role] is not None else None) for role in ROLE_NAMES
    }
    audit = []
    requested = indices if query_indices is None else query_indices
    if not requested or any(index not in indices for index in requested):
        raise ValueError("Sparse grounding indices must be a non-empty subset of video frames")
    for start in range(0, len(requested), tracking_chunk_size):
        current = requested[start : start + tracking_chunk_size]
        prompt = tracking_prompt(goal, roles, current)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index in current:
            content.extend([
                {"type": "text", "text": f"frame {index}"},
                {"type": "image_url", "image_url": {"url": data_url(output / "frames" / f"frame_{index:06d}.png")}},
            ])
        chunk_tracks, attempts = client.request(
            content,
            max_tokens,
            lambda value, requested=current: validate_tracking_document(value, requested, roles),
        )
        for role in ROLE_NAMES:
            if frame_grounding[role] is not None:
                frame_grounding[role].extend(chunk_tracks[role] or [])
        audit.append({"frames": current, "attempts": attempts})
    for role in ROLE_NAMES:
        if frame_grounding[role] is None:
            continue
        by_index = {row["frame_index"]: row for row in frame_grounding[role]}
        frame_grounding[role] = [
            by_index.get(index, {"frame_index": index, "bbox_xyxy_1000": None})
            for index in indices
        ]
    return frame_grounding, audit


def infer_entities(output: Path, config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    target = output / "task_spec.json"
    fingerprint = stage_fingerprint(config, [output / "input.json"], "entities")
    if not overwrite and stage_current(output, "entities", fingerprint, [target]):
        return read_json(target)
    context = read_json(output / "input.json")
    client = QwenClient(config)
    include_broad_concepts = bool(config["qwen"].get("agent_concept_pairs_enabled", False))
    indices = list(range(int(context["frame_count"])))
    documents = []
    audit_attempts = []
    for current, previous in contextual_chunks(indices, int(config["qwen"]["frames_per_request"])):
        visual_indices = ([previous] if previous is not None else []) + current
        document, attempts = client.request(
            entity_content(
                context["planning_goal"], output / "frames", visual_indices,
                include_broad_concepts,
            ),
            int(config["qwen"]["max_tokens_entities"]),
            lambda value: validate_entity_document(value, include_broad_concepts),
        )
        documents.append(document)
        audit_attempts.append({"frames": visual_indices, "attempts": attempts})
    if len(documents) == 1:
        merged = documents[0]
    else:
        reconciliation = [{"type": "text", "text": (
            "Reconcile these chunk-level entity descriptions for one video into the exact same roles/task_actions JSON schema. "
            "Prefer stable visually specific names; robot and manipulated_object are required. Planning goal: "
            + context["planning_goal"] + "\nCandidates:\n" + json.dumps(documents, ensure_ascii=False)
        )}]
        merged, attempts = client.request(
            reconciliation,
            int(config["qwen"]["max_tokens_entities"]),
            lambda value: validate_entity_document(value, include_broad_concepts),
        )
        audit_attempts.append({"stage": "reconcile", "attempts": attempts})
    initial_proposal = json.loads(json.dumps(merged))
    frame_grounding_enabled = bool(config["qwen"].get("frame_grounding_enabled", True))
    frame_grounding: dict[str, list[dict[str, Any]] | None] = {
        role: None for role in ROLE_NAMES
    }
    tracking_audit = []
    if frame_grounding_enabled:
        tracking_chunk_size = int(config["qwen"]["tracking_frames_per_request"])
        if not 1 <= tracking_chunk_size <= int(config["qwen"]["frames_per_request"]):
            raise ValueError("tracking_frames_per_request must be between 1 and frames_per_request")
        grounding_indices = sparse_anchor_indices(
            indices,
            int(config["qwen"].get("box_anchor_count_per_window", len(indices))),
            int(config["qwen"]["frames_per_request"]),
        )
        frame_grounding, tracking_audit = ground_roles_in_chunks(
            client,
            context["planning_goal"],
            merged["roles"],
            output,
            indices,
            tracking_chunk_size,
            int(config["qwen"]["max_tokens_tracking"]),
            grounding_indices,
        )
    else:
        grounding_indices = []
    initial_frame_grounding = json.loads(json.dumps(frame_grounding))
    reflection_audit = []
    initial_tracking_audit = tracking_audit
    if bool(config["qwen"].get("entity_reflection_enabled", False)):
        if not frame_grounding_enabled:
            raise ValueError("Box-overlay entity reflection requires frame grounding")
        reflected_documents = []
        for current, previous in contextual_chunks(indices, int(config["qwen"]["frames_per_request"])):
            visual_indices = ([previous] if previous is not None else []) + current
            reflected, attempts = client.request(
                entity_reflection_content(
                    context["planning_goal"], output, merged, frame_grounding, visual_indices,
                ),
                int(config["qwen"].get("max_tokens_entity_reflection", config["qwen"]["max_tokens_entities"])),
                validate_entity_document,
            )
            reflected_documents.append(reflected)
            reflection_audit.append({"frames": visual_indices, "attempts": attempts})
        if len(reflected_documents) == 1:
            merged = reflected_documents[0]
        else:
            reconciliation = [{"type": "text", "text": (
                "Reconcile these reflection-stage entity proposals into the exact roles/task_actions JSON schema. "
                "Use stable identities supported across the whole video and do not add extra fields. Planning goal: "
                + context["planning_goal"] + "\nCandidates:\n"
                + json.dumps(reflected_documents, ensure_ascii=False)
            )}]
            merged, attempts = client.request(
                reconciliation,
                int(config["qwen"].get("max_tokens_entity_reflection", config["qwen"]["max_tokens_entities"])),
                validate_entity_document,
            )
            reflection_audit.append({"stage": "reconcile", "attempts": attempts})
        frame_grounding, tracking_audit = ground_roles_in_chunks(
            client,
            context["planning_goal"],
            merged["roles"],
            output,
            indices,
            tracking_chunk_size,
            int(config["qwen"]["max_tokens_tracking"]),
            grounding_indices,
        )
    entities = []
    role_map = {}
    kinds = {
        "robot": "robot", "manipulated_object": "object", "initial_support": "support_surface",
        "target": "object", "whole_parent": "whole_parent",
    }
    for role in ROLE_NAMES:
        entity = merged["roles"][role]
        role_map[role] = role if entity else None
        if entity:
            entity_record = {
                "entity_id": role,
                "role": role,
                "entity_kind": kinds[role],
                **entity,
            }
            if frame_grounding_enabled:
                entity_record["frame_grounding"] = frame_grounding[role]
            entities.append(entity_record)
    result = {
        "schema_version": "unified_sgg_task_spec_v1",
        "planning_goal": context["planning_goal"],
        "entities": entities,
        "roles": role_map,
        "task_actions": sorted(set(merged["task_actions"])),
        "provenance": {
            "model": config["qwen"]["model"],
            "model_revision": config["qwen"]["revision"],
            "prompt_version": config["qwen"]["entity_prompt_version"],
            "tracking_prompt_version": config["qwen"]["tracking_prompt_version"],
            "frame_grounding_enabled": frame_grounding_enabled,
            "entity_reflection_enabled": bool(config["qwen"].get("entity_reflection_enabled", False)),
            "entity_reflection_prompt_version": config["qwen"].get("entity_reflection_prompt_version"),
            "config_hash": config_hash(config),
            "all_frames_used": True,
            "all_frames_tracked": False,
            "box_anchor_strategy": config["qwen"].get("box_anchor_strategy"),
            "box_anchor_indices": grounding_indices,
            "box_anchor_count": len(grounding_indices),
        },
    }
    write_json_atomic(target, result)
    write_json_atomic(output / "qwen_entities_audit.json", {
        "prompt_template": entity_prompt(
            context["planning_goal"], ["<ordered_frame_indices>"], include_broad_concepts,
        ),
        "chunks": audit_attempts,
        "initial_proposal": initial_proposal,
        "initial_frame_grounding": initial_frame_grounding,
        "initial_tracking_chunks": initial_tracking_audit,
        "reflection_prompt_template": (
            entity_reflection_prompt(
                context["planning_goal"], initial_proposal, ["<ordered_frame_indices>"],
            )
            if frame_grounding_enabled else None
        ),
        "reflection_chunks": reflection_audit,
        "tracking_prompt_template": (
            tracking_prompt(
                context["planning_goal"], merged["roles"], ["<ordered_frame_indices>"],
            )
            if frame_grounding_enabled else None
        ),
        "tracking_chunks": tracking_audit,
        "box_anchor_indices": grounding_indices,
        "box_anchor_count": len(grounding_indices),
    })
    complete_stage(output, "entities", fingerprint)
    update_run_report(output, "entities", "success", {"entity_count": len(entities)})
    return result
