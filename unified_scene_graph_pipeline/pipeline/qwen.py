from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests

from common import (
    complete_stage, config_hash, mask_geometry, read_json, require_api_key,
    stage_current, stage_fingerprint, update_run_report, write_json_atomic,
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


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


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
        for attempt in range(int(self.config["retries"]) + 1):
            started = time.monotonic()
            payload = {
                "model": self.config["model"],
                "messages": [{"role": "user", "content": content}],
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
                content = [*content, {"type": "text", "text": f"Your prior JSON was invalid: {error}. Return corrected complete JSON only."}]
        raise RuntimeError(f"Qwen schema failed after {len(attempts)} attempts: {attempts[-1].get('validation_error')}")


def entity_prompt(goal: str, frame_indices: list[int]) -> str:
    return f"""You identify task entities in ordered robot-manipulation frames.
Planning goal: {goal}
The following images are frames {frame_indices} in temporal order.
Use visual appearance, not hidden annotations. Return exactly this JSON shape:
{{"roles":{{
 "robot":{{"canonical_name":"specific visible name","sam_prompt":"robot arm","grounding":{{"frame_index":<supplied integer>,"bbox_xyxy_1000":[xmin,ymin,xmax,ymax]}}}},
 "manipulated_object":{{"canonical_name":"specific visible name","sam_prompt":"short visual description","grounding":{{"frame_index":<supplied integer>,"bbox_xyxy_1000":[xmin,ymin,xmax,ymax]}}}},
 "initial_support":null or {{"canonical_name":"...","sam_prompt":"...","grounding":{{"frame_index":<supplied integer>,"bbox_xyxy_1000":[xmin,ymin,xmax,ymax]}}}},
 "target":null or {{"canonical_name":"...","sam_prompt":"...","grounding":{{"frame_index":<supplied integer>,"bbox_xyxy_1000":[xmin,ymin,xmax,ymax]}}}},
 "whole_parent":null or {{"canonical_name":"...","sam_prompt":"...","grounding":{{"frame_index":<supplied integer>,"bbox_xyxy_1000":[xmin,ymin,xmax,ymax]}}}}
}},"task_actions":["canonical action labels"]}}
Allowed task actions: reach_for, grab, lift, move, place, release, push, pour, open, close, insert, stack.
Robot and manipulated_object must be present. Identify the manipulated object as the physical object whose position or state changes across the ordered frames, using the goal only as context. Initial support is the visible surface or receptacle supporting the manipulated object in the early frames. Target is the visible intended destination surface or receptacle, supported by the goal and the observed motion or final frames. If the goal conflicts with the visible action, trust the video; use null for a target that is neither visible nor supported by the observed action.

For each non-null role, choose exactly one supplied frame where that single entity is clearest and least occluded. Return a tight bounding box around only that entity. bbox_xyxy_1000 is [left, top, right, bottom] in normalized integer coordinates: the top-left image corner is [0,0] and the bottom-right is [1000,1000]. The frame_index must be one of the supplied frame labels. Do not box a collection when the role refers to one manipulated object, and do not include nearby objects or robot parts when avoidable.

Keep visually specific short descriptions: robot sam_prompt must be exactly "robot arm"; other sam_prompt values should normally be a discriminative color plus a common object noun, such as "yellow banana", "brown cup", "red cube", "black clamp", "wooden table", "orange bowl", "silver plate", or "blue container". canonical_name may remain more specific. Do not output points, masks, confidence, planning steps, synonyms, or extra roles."""


def validate_entity_document(
    value: dict[str, Any], allowed_frame_indices: set[int] | None = None,
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
        if not isinstance(entity, dict) or set(entity) != {"canonical_name", "sam_prompt", "grounding"}:
            raise ValueError(f"Invalid entity shape for {role}")
        if not all(isinstance(entity[key], str) and entity[key].strip() for key in ("canonical_name", "sam_prompt")):
            raise ValueError(f"Empty entity value for {role}")
        grounding = entity["grounding"]
        if not isinstance(grounding, dict) or set(grounding) != {"frame_index", "bbox_xyxy_1000"}:
            raise ValueError(f"Invalid grounding shape for {role}")
        frame_index = grounding["frame_index"]
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError(f"Invalid grounding frame for {role}")
        if allowed_frame_indices is not None and frame_index not in allowed_frame_indices:
            raise ValueError(f"Grounding frame for {role} was not supplied")
        box = grounding["bbox_xyxy_1000"]
        if (
            not isinstance(box, list) or len(box) != 4
            or any(not isinstance(coordinate, int) or isinstance(coordinate, bool) for coordinate in box)
            or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)
        ):
            raise ValueError(f"Invalid normalized grounding box for {role}")
    allowed = {"reach_for", "grab", "lift", "move", "place", "release", "push", "pour", "open", "close", "insert", "stack"}
    actions = value["task_actions"]
    if not isinstance(actions, list) or any(item not in allowed for item in actions):
        raise ValueError("Invalid task action")
    # This role-level normalization is part of the shared pipeline contract. It is
    # deliberately independent of the dataset, episode, and Qwen wording.
    roles["robot"]["sam_prompt"] = "robot arm"
    return value


def entity_content(goal: str, frames: Path, indices: list[int]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": entity_prompt(goal, indices)}]
    for index in indices:
        content.extend([
            {"type": "text", "text": f"frame {index}"},
            {"type": "image_url", "image_url": {"url": data_url(frames / f"frame_{index:06d}.png")}},
        ])
    return content


def infer_entities(output: Path, config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    target = output / "task_spec.json"
    fingerprint = stage_fingerprint(config, [output / "input.json"], "entities")
    if not overwrite and stage_current(output, "entities", fingerprint, [target]):
        return read_json(target)
    context = read_json(output / "input.json")
    client = QwenClient(config)
    indices = list(range(int(context["frame_count"])))
    documents = []
    audit_attempts = []
    for current, previous in contextual_chunks(indices, int(config["qwen"]["frames_per_request"])):
        visual_indices = ([previous] if previous is not None else []) + current
        document, attempts = client.request(
            entity_content(context["planning_goal"], output / "frames", visual_indices),
            int(config["qwen"]["max_tokens_entities"]),
            lambda value, allowed=set(visual_indices): validate_entity_document(value, allowed),
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
            lambda value: validate_entity_document(value, set(indices)),
        )
        audit_attempts.append({"stage": "reconcile", "attempts": attempts})
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
            entities.append({"entity_id": role, "role": role, "entity_kind": kinds[role], **entity})
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
            "config_hash": config_hash(config),
            "all_frames_used": True,
        },
    }
    write_json_atomic(target, result)
    write_json_atomic(output / "qwen_entities_audit.json", {
        "prompt_template": entity_prompt(context["planning_goal"], ["<ordered_frame_indices>"]),
        "chunks": audit_attempts,
    })
    complete_stage(output, "entities", fingerprint)
    update_run_report(output, "entities", "success", {"entity_count": len(entities)})
    return result


def graph_context(goal: str, task: dict[str, Any], frame_indices: list[int]) -> tuple[list[str], dict[str, dict[str, str]], str]:
    entity_ids = [entity["entity_id"] for entity in task["entities"]]
    mask_legend = {
        entity["entity_id"]: {
            "color": MASK_COLORS[entity["role"]][1],
            "marker": MASK_MARKERS[entity["role"]],
        }
        for entity in task["entities"]
    }
    graph_entities = [
        {key: value for key, value in entity.items() if key != "grounding"}
        for entity in task["entities"]
    ]
    context = f"""Planning goal: {goal}
Entities: {json.dumps(graph_entities, ensure_ascii=False)}
Frames to output: {frame_indices}
Each image is the RGB frame with transparent mask overlays. Mask color legend: {json.dumps(mask_legend)}. Use the image itself, masks, goal, and role specification only."""
    return entity_ids, mask_legend, context


def state_graph_prompt(goal: str, task: dict[str, Any], frame_indices: list[int], config: dict[str, Any]) -> str:
    entity_ids, _, context = graph_context(goal, task, frame_indices)
    return f"""Infer visible state relations independently for every supplied robot-manipulation frame.
{context}
Return compact JSON only: {{"frames":[[frame_index,[[subject,relation,object]]], ...]}}.
Return one item for each requested frame, and no others. Valid entity IDs: {entity_ids}.
Allowed state relations: {config['ontology']['states']}.
State edges describe only what is visibly true in that frame. Common examples are ["manipulated_object","on","initial_support"], ["robot","holding","manipulated_object"], and ["manipulated_object","inside","target"]. Pay particular attention to the first and final frames: a completed manipulation normally removes the initial-support relation and adds the target relation. A mask can be absent because of occlusion, so use RGB evidence as well as overlays, but never invent an entity.
Infer every requested frame directly rather than copying the prior frame. Edge object may be null only for unary open or closed. Omit unsupported edges. Do not output actions, confidence, reasoning, coordinates, descriptions, planning steps, or extra fields."""


def action_graph_prompt(goal: str, task: dict[str, Any], frame_indices: list[int], config: dict[str, Any]) -> str:
    entity_ids, _, context = graph_context(goal, task, frame_indices)
    return f"""Infer temporally supported robot actions for every supplied robot-manipulation frame.
{context}
Return compact JSON only: {{"frames":[[frame_index,[[actor,action,object_or_null]]], ...]}}.
Return one item for each requested frame, and no others. Valid entity IDs: {entity_ids}.
Allowed actions: {config['ontology']['actions']}.
Use the complete ordered sequence to recognize transitions and timing. reach_for occurs only during approach; grab at grasp closure; lift when the object leaves its support; move during transport while held; place at target contact; release when the gripper lets go. Use push, pour, open, close, insert, or stack only when the corresponding motion is visible. Many frames may have no active action, and an unsuccessful video may never complete the planned action.
Do not infer static state relations in this pass. Omit unsupported actions. Do not output confidence, reasoning, coordinates, descriptions, planning steps, or extra fields."""


def overlay_frame(output: Path, task: dict[str, Any], frame_index: int) -> bytes:
    from io import BytesIO
    from PIL import Image, ImageDraw
    import numpy as np

    frame = Image.open(output / "frames" / f"frame_{frame_index:06d}.png").convert("RGB")
    pixels = np.asarray(frame).copy()
    markers = []
    for entity in task["entities"]:
        mask_path = output / "masks" / entity["entity_id"] / f"frame_{frame_index:06d}.png"
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.any():
            color = np.asarray(MASK_COLORS[entity["role"]][0])
            pixels[mask] = (0.62 * pixels[mask] + 0.38 * color).astype(np.uint8)
            ys, xs = np.nonzero(mask)
            markers.append((int(xs.min()), int(ys.min()), MASK_MARKERS[entity["role"]], tuple(color.tolist())))
    rendered = Image.fromarray(pixels)
    draw = ImageDraw.Draw(rendered)
    for x, y, marker, color in markers:
        box = draw.textbbox((x, y), marker)
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((x, y), marker, fill=color)
    stream = BytesIO()
    rendered.save(stream, format="JPEG", quality=88)
    return stream.getvalue()


def graph_content(output: Path, prompt: str, task: dict[str, Any], indices: list[int]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index in indices:
        encoded = base64.b64encode(overlay_frame(output, task, index)).decode("ascii")
        content.extend([
            {"type": "text", "text": f"frame {index}"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
        ])
    return content


def validate_state_document(value: dict[str, Any], requested: list[int], task: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    if set(value) != {"frames"} or not isinstance(value["frames"], list):
        raise ValueError("Expected only frames list")
    packed_rows = value["frames"]
    if any(not isinstance(row, list) or len(row) != 2 for row in packed_rows):
        raise ValueError("Each state frame must be [frame_index, state_edges]")
    if sorted(row[0] for row in packed_rows) != sorted(requested) or len(packed_rows) != len(requested):
        raise ValueError(f"Expected exactly frame indices {requested}")
    entity_ids = {entity["entity_id"] for entity in task["entities"]}
    states = set(config["ontology"]["states"])
    rows = []
    for frame_index, packed_edges in packed_rows:
        if not isinstance(packed_edges, list):
            raise ValueError("State edges must be a list")
        state_edges = []
        for packed_edge in packed_edges:
            if not isinstance(packed_edge, list) or len(packed_edge) != 3:
                raise ValueError(f"Invalid compact state edge {packed_edge}")
            edge = dict(zip(("subject", "relation", "object"), packed_edge))
            unary_valid = edge.get("relation") in {"open", "closed"} and edge.get("object") is None
            binary_valid = edge.get("object") in entity_ids
            if edge["subject"] not in entity_ids or edge["relation"] not in states or not (unary_valid or binary_valid):
                raise ValueError(f"Invalid state edge {edge}")
            state_edges.append(edge)
        rows.append({"frame_index": frame_index, "state_edges": state_edges})
    return rows


def validate_action_document(value: dict[str, Any], requested: list[int], task: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    if set(value) != {"frames"} or not isinstance(value["frames"], list):
        raise ValueError("Expected only frames list")
    packed_rows = value["frames"]
    if any(not isinstance(row, list) or len(row) != 2 for row in packed_rows):
        raise ValueError("Each action frame must be [frame_index, actions]")
    if sorted(row[0] for row in packed_rows) != sorted(requested) or len(packed_rows) != len(requested):
        raise ValueError(f"Expected exactly frame indices {requested}")
    entity_ids = {entity["entity_id"] for entity in task["entities"]}
    actions = set(config["ontology"]["actions"])
    rows = []
    for frame_index, packed_actions in packed_rows:
        if not isinstance(packed_actions, list):
            raise ValueError("Actions must be a list")
        expanded_actions = []
        for packed_action in packed_actions:
            if not isinstance(packed_action, list) or len(packed_action) != 3:
                raise ValueError(f"Invalid compact action {packed_action}")
            action = dict(zip(("actor", "action", "object"), packed_action))
            if action["actor"] not in entity_ids or action["action"] not in actions or (action["object"] is not None and action["object"] not in entity_ids):
                raise ValueError(f"Invalid action {action}")
            expanded_actions.append(action)
        rows.append({"frame_index": frame_index, "actions": expanded_actions})
    return rows


def infer_graph(output: Path, config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    dependencies = [output / "input.json", output / "task_spec.json", output / "tracks.json", output / "masks"]
    fingerprint = stage_fingerprint(config, dependencies, "graph")
    target = output / "scene_graph.json"
    if not overwrite and stage_current(output, "graph", fingerprint, [target]):
        return read_json(target)
    context = read_json(output / "input.json")
    task = read_json(output / "task_spec.json")
    client = QwenClient(config)
    indices = list(range(int(context["frame_count"])))
    state_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    state_audit = []
    action_audit = []
    for current, previous in contextual_chunks(indices, int(config["qwen"]["frames_per_request"])):
        # A previous RGB+mask image provides visual boundary context only; it is explicitly excluded from output.
        visual = ([previous] if previous is not None else []) + current
        state_prompt = state_graph_prompt(context["planning_goal"], task, current, config) + (f"\nFrame {previous} is context only and MUST NOT appear in output." if previous is not None else "")
        chunk_states, attempts = client.request(
            graph_content(output, state_prompt, task, visual),
            int(config["qwen"]["max_tokens_graph"]),
            lambda value, requested=current: validate_state_document(value, requested, task, config),
        )
        state_rows.extend(chunk_states)
        state_audit.append({"output_frames": current, "visual_frames": visual, "attempts": attempts})
        action_prompt = action_graph_prompt(context["planning_goal"], task, current, config) + (f"\nFrame {previous} is context only and MUST NOT appear in output." if previous is not None else "")
        chunk_actions, attempts = client.request(
            graph_content(output, action_prompt, task, visual),
            int(config["qwen"]["max_tokens_graph"]),
            lambda value, requested=current: validate_action_document(value, requested, task, config),
        )
        action_rows.extend(chunk_actions)
        action_audit.append({"output_frames": current, "visual_frames": visual, "attempts": attempts})
    state_by_index = {row["frame_index"]: row["state_edges"] for row in state_rows}
    action_by_index = {row["frame_index"]: row["actions"] for row in action_rows}
    frames = []
    for index in indices:
        nodes = []
        for entity in task["entities"]:
            geometry = mask_geometry(output / "masks" / entity["entity_id"] / f"frame_{index:06d}.png")
            nodes.append({"entity_id": entity["entity_id"], "role": entity["role"], "canonical_name": entity["canonical_name"], **geometry})
        frames.append({"frame_index": index, "nodes": nodes, "state_edges": state_by_index[index], "actions": action_by_index[index]})
    result = {
        "schema_version": "unified_scene_graph_v1",
        "frame_count": len(frames),
        "frames": frames,
        "ontology": config["ontology"],
        "provenance": {"model": config["qwen"]["model"], "model_revision": config["qwen"]["revision"], "prompt_version": config["qwen"]["graph_prompt_version"], "config_hash": config_hash(config), "per_frame_inference": True, "split_state_action_inference": True},
    }
    write_json_atomic(target, result)
    write_json_atomic(output / "qwen_graph_audit.json", {
        "state_prompt_template": state_graph_prompt(context["planning_goal"], task, ["<output_frame_indices>"], config),
        "action_prompt_template": action_graph_prompt(context["planning_goal"], task, ["<output_frame_indices>"], config),
        "state_chunks": state_audit,
        "action_chunks": action_audit,
    })
    complete_stage(output, "graph", fingerprint, {"frame_count": len(frames)})
    update_run_report(output, "graph", "success", {"frame_count": len(frames)})
    return result
