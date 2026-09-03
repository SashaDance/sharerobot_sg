from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))

from common import ensure_no_confidence, image_paths  # noqa: E402
from prepare import prepare  # noqa: E402
from qwen import (  # noqa: E402
    contextual_chunks,
    entity_reflection_prompt,
    expand_event_graph,
    tracking_overlay_frame,
    sparse_anchor_indices,
    validate_action_document,
    validate_entity_document,
    validate_event_graph_document,
    validate_state_document,
    validate_tracking_document,
)
from render import ROLE_COLORS, _draw_frame  # noqa: E402
from robot_tracking_compare import _config as robot_tracking_config, _diagnostics  # noqa: E402
from sam_stage import (  # noqa: E402
    _candidate_overlay,
    _candidate_text_prompts,
    _prefer_primary_masks,
    _qwen_box_chunks,
    _validate_candidate_selection,
)


def test_role_specific_candidate_prompts_are_global_and_deduplicated() -> None:
    entity = {
        "role": "manipulated_object",
        "canonical_name": "red cube",
        "sam_prompt": "small red cube",
    }
    config = {
        "candidate_prompts": ["sam_prompt", "canonical_name"],
        "candidate_prompt_templates": {
            "manipulated_object": [
                "{sam_prompt}",
                "{canonical_name}",
                "the individual {canonical_name} manipulated by the robot",
                "{canonical_name}",
            ]
        },
    }
    assert _candidate_text_prompts(entity, config, "stack a red cube") == [
        "small red cube",
        "red cube",
        "the individual red cube manipulated by the robot",
    ]
    support = {"role": "target", "canonical_name": "green cube", "sam_prompt": "green cube"}
    assert _candidate_text_prompts(support, config, "stack a red cube") == ["green cube"]


def test_prepare_preserves_all_arbitrary_length_rgba_frames(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(7):
        Image.new("RGBA", (23, 17), (index, 20, 30, 120)).save(source / f"frame_{index}.png")
    goal = tmp_path / "goal.json"
    goal.write_text(json.dumps({"schema_version": "1.0", "planning_goal": "move the blue cube", "source_dataset": "test"}))
    output = tmp_path / "output"
    result = prepare(output, source, None, goal, None)
    assert result["frame_count"] == 7
    assert [path.name for path in image_paths(output / "frames")] == [f"frame_{index:06d}.png" for index in range(7)]
    assert all(Image.open(path).mode == "RGB" for path in image_paths(output / "frames"))


def test_entity_schema_rejects_extra_roles_and_confidence() -> None:
    valid = {
        "roles": {
            "robot": {
                "canonical_name": "gray robot arm", "sam_prompt": "gray robot arm",
            },
            "manipulated_object": {
                "canonical_name": "blue cube", "sam_prompt": "small blue cube",
            },
            "initial_support": None, "target": None, "whole_parent": None,
        },
        "task_actions": ["move"],
    }
    assert validate_entity_document(valid) == valid
    assert valid["roles"]["robot"]["sam_prompt"] == "robot arm"
    invalid = json.loads(json.dumps(valid))
    invalid["roles"]["tool"] = None
    try:
        validate_entity_document(invalid)
        raise AssertionError("extra role accepted")
    except ValueError:
        pass
    invalid_tracking = {
        "tracks": {
            "robot": [[0, [100, 50, 900, 950]], [1, None]],
            "manipulated_object": [[0, [300, 300, 450, 500]], [1, [320, 300, 470, 500]]],
            "initial_support": None,
            "target": None,
            "whole_parent": None,
        }
    }
    normalized = validate_tracking_document(invalid_tracking, [0, 1], valid["roles"])
    assert normalized["robot"][1]["bbox_xyxy_1000"] is None
    invalid_tracking["tracks"]["manipulated_object"][1][1] = [450, 300, 300, 500]
    try:
        validate_tracking_document(invalid_tracking, [0, 1], valid["roles"])
        raise AssertionError("invalid grounding box accepted")
    except ValueError:
        pass
    try:
        ensure_no_confidence({"confidence": 0.9})
        raise AssertionError("confidence accepted")
    except ValueError:
        pass


def test_agent_entity_schema_requires_core_and_broad_concepts() -> None:
    value = {
        "roles": {
            "robot": {
                "canonical_name": "gray robot arm",
                "sam_prompt": "robot arm",
                "broad_sam_prompt": "robot",
            },
            "manipulated_object": {
                "canonical_name": "left black grid clamp",
                "sam_prompt": "black clamp",
                "broad_sam_prompt": "clamp",
            },
            "initial_support": None,
            "target": None,
            "whole_parent": None,
        },
        "task_actions": ["move"],
    }
    assert validate_entity_document(value, include_broad_concepts=True) == value
    del value["roles"]["manipulated_object"]["broad_sam_prompt"]
    try:
        validate_entity_document(value, include_broad_concepts=True)
        raise AssertionError("missing broad concept accepted")
    except ValueError:
        pass


def test_agent_candidate_decisions_are_complete_and_selected_is_not_rejected() -> None:
    value = {
        "decisions": [
            {"candidate_id": "C0", "decision": "rejected"},
            {"candidate_id": "C1", "decision": "accepted"},
        ],
        "selected_candidate_id": "C1",
    }
    assert _validate_candidate_selection(value, ["C0", "C1"]) == value
    value["selected_candidate_id"] = "C0"
    try:
        _validate_candidate_selection(value, ["C0", "C1"])
        raise AssertionError("rejected candidate selected")
    except ValueError:
        pass


def test_agent_candidate_overlay_marks_every_visible_track(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (40, 30), (100, 100, 100)).save(frame)
    first = np.zeros((30, 40), dtype=bool)
    second = np.zeros((30, 40), dtype=bool)
    first[5:12, 4:11] = True
    second[16:25, 22:35] = True
    payload = _candidate_overlay(
        frame,
        [{"masks": {0: first}}, {"masks": {0: second}}],
        0,
    )
    assert payload.startswith(b"\x89PNG")
    assert len(payload) > frame.stat().st_size


def test_graph_schema_requires_requested_frames_and_valid_references() -> None:
    task = {"entities": [{"entity_id": "robot"}, {"entity_id": "manipulated_object"}]}
    config = {"ontology": {"states": ["holding"], "actions": ["grab"]}}
    states = {"frames": [[4, [["robot", "holding", "manipulated_object"]]]]}
    actions = {"frames": [[4, [["robot", "grab", "manipulated_object"]]]]}
    assert validate_state_document(states, [4], task, config)[0]["state_edges"][0]["relation"] == "holding"
    assert validate_action_document(actions, [4], task, config)[0]["actions"][0]["action"] == "grab"
    states["frames"][0][1][0][2] = "unknown"
    try:
        validate_state_document(states, [4], task, config)
        raise AssertionError("invalid reference accepted")
    except ValueError:
        pass


def test_context_chunks_never_exceed_thirty_images() -> None:
    chunks = list(contextual_chunks(list(range(117)), 30))
    assert [index for current, _ in chunks for index in current] == list(range(117))
    assert all(len(current) + (previous is not None) <= 30 for current, previous in chunks)
    assert all(previous == current[0] - 1 for current, previous in chunks[1:])


def test_sparse_box_anchors_cover_each_thirty_frame_window() -> None:
    assert sparse_anchor_indices(list(range(30)), 6, 30) == [0, 6, 12, 18, 24, 29]
    anchors = sparse_anchor_indices(list(range(67)), 6, 30)
    assert anchors == [0, 6, 12, 18, 24, 29, 30, 36, 42, 48, 54, 59, 60, 62, 63, 64, 65, 66]
    assert anchors[0] == 0 and anchors[-1] == 66


def test_entity_reflection_uses_all_roles_without_confidence(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (100, 80), (30, 40, 50)).save(frame)
    roles = {
        "robot": {"canonical_name": "robot arm", "sam_prompt": "robot arm"},
        "manipulated_object": {"canonical_name": "blue cube", "sam_prompt": "blue cube"},
        "initial_support": None,
        "target": None,
        "whole_parent": None,
    }
    grounding = {
        "robot": [{"frame_index": 0, "bbox_xyxy_1000": [0, 0, 500, 500]}],
        "manipulated_object": [{"frame_index": 0, "bbox_xyxy_1000": [500, 500, 900, 900]}],
        "initial_support": None,
        "target": None,
        "whole_parent": None,
    }
    rendered = tracking_overlay_frame(frame, roles, grounding, 0)
    assert rendered.startswith(b"\x89PNG")
    prompt = entity_reflection_prompt(
        "move the blue cube", {"roles": roles, "task_actions": ["move"]}, [0],
    )
    assert "complete visible trajectory" in prompt
    assert set(roles) == {"robot", "manipulated_object", "initial_support", "target", "whole_parent"}


def test_qwen_box_chunks_preserve_all_frames_and_null_gaps() -> None:
    rows = [
        {"frame_index": index, "bbox_xyxy_1000": None if index in {1, 5} else [1, 2, 3, 4]}
        for index in range(7)
    ]
    chunks = _qwen_box_chunks(rows, frame_count=7, chunk_size=3)
    assert [(start, end) for start, end, _ in chunks] == [(0, 3), (3, 6), (6, 7)]
    assert [[row["frame_index"] for row in grounded] for _, _, grounded in chunks] == [
        [0, 2], [3, 4], [6],
    ]


def test_qwen_box_chunks_reject_incomplete_grounding() -> None:
    try:
        _qwen_box_chunks(
            [{"frame_index": 0, "bbox_xyxy_1000": [1, 2, 3, 4]}],
            frame_count=2,
            chunk_size=5,
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("incomplete frame grounding accepted")


def test_hybrid_masks_never_replace_primary_frames() -> None:
    primary = {0: np.asarray([[True, False]]), 2: np.asarray([[False, True]])}
    fallback = {0: np.asarray([[False, True]]), 1: np.asarray([[True, True]])}
    merged, fallback_frames = _prefer_primary_masks(primary, fallback)
    assert np.array_equal(merged[0], primary[0])
    assert np.array_equal(merged[1], fallback[1])
    assert np.array_equal(merged[2], primary[2])
    assert fallback_frames == [1]


def test_graph_schema_accepts_honest_empty_or_unchanged_chunks() -> None:
    task = {
        "entities": [{"entity_id": "robot"}, {"entity_id": "manipulated_object"}],
        "task_actions": ["move"],
    }
    config = {"ontology": {"states": ["holding"], "actions": ["move"]}}
    empty = {"frames": [[0, []], [1, []]]}
    assert validate_state_document(empty, [0, 1], task, config) == [
        {"frame_index": 0, "state_edges": []},
        {"frame_index": 1, "state_edges": []},
    ]
    assert validate_action_document(empty, [0, 1], task, config) == [
        {"frame_index": 0, "actions": []},
        {"frame_index": 1, "actions": []},
    ]


def test_event_graph_replays_persistent_state_and_action_intervals() -> None:
    task = {
        "entities": [{"entity_id": "robot"}, {"entity_id": "manipulated_object"}, {"entity_id": "target"}],
    }
    config = {"ontology": {"states": ["on", "holding"], "actions": ["grab", "move", "place"]}}
    document = {
        "initial_state": [["robot", "holding", "manipulated_object"]],
        "events": [[3, 4, "robot", "move", "manipulated_object"], [5, 5, "robot", "place", "manipulated_object"]],
        "transitions": [[5, [["robot", "holding", "manipulated_object"]], [["manipulated_object", "on", "target"]]]],
        "final_state": [["manipulated_object", "on", "target"]],
    }
    verified = validate_event_graph_document(document, [3, 4, 5], task, config)
    states, actions = expand_event_graph(verified, [3, 4, 5])
    assert states[0]["state_edges"] == [{"subject": "robot", "relation": "holding", "object": "manipulated_object"}]
    assert states[-1]["state_edges"] == [{"subject": "manipulated_object", "relation": "on", "object": "target"}]
    assert [item["actions"][0]["action"] for item in actions] == ["move", "move", "place"]


def test_event_graph_rejects_inconsistent_final_state() -> None:
    task = {"entities": [{"entity_id": "robot"}, {"entity_id": "manipulated_object"}]}
    config = {"ontology": {"states": ["holding"], "actions": ["grab"]}}
    document = {
        "initial_state": [],
        "events": [],
        "transitions": [],
        "final_state": [["robot", "holding", "manipulated_object"]],
    }
    try:
        validate_event_graph_document(document, [0, 1], task, config)
    except ValueError as error:
        assert "final_state" in str(error)
    else:
        raise AssertionError("inconsistent final state accepted")


def test_event_graph_enforces_state_continuity_between_chunks() -> None:
    task = {"entities": [{"entity_id": "robot"}, {"entity_id": "manipulated_object"}]}
    config = {"ontology": {"states": ["holding"], "actions": ["move"]}}
    document = {
        "initial_state": [],
        "events": [],
        "transitions": [],
        "final_state": [],
    }
    try:
        validate_event_graph_document(
            document,
            [30, 31],
            task,
            config,
            [["robot", "holding", "manipulated_object"]],
        )
    except ValueError as error:
        assert "prior chunk" in str(error)
    else:
        raise AssertionError("cross-chunk state discontinuity accepted")


def test_render_has_prompt_legend_and_strong_mask_boundary(tmp_path: Path) -> None:
    output = tmp_path / "scene"
    (output / "frames").mkdir(parents=True)
    (output / "masks" / "manipulated_object").mkdir(parents=True)
    Image.new("RGB", (80, 60), (120, 120, 120)).save(output / "frames" / "frame_000000.png")
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[20:40, 25:50] = 255
    Image.fromarray(mask).save(output / "masks" / "manipulated_object" / "frame_000000.png")
    task = {
        "planning_goal": "move the blue cube",
        "entities": [{
            "entity_id": "manipulated_object", "role": "manipulated_object",
            "canonical_name": "blue cube", "sam_prompt": "small blue cube",
        }],
    }
    graph = {"frames": [{"state_edges": [], "actions": []}]}
    rendered = _draw_frame(
        output, 0, task, graph,
        {"min_width": 80, "mask_alpha": 0.48, "boundary_width": 2},
    )
    assert rendered.width == 80
    assert rendered.height > 60
    pixels = np.asarray(rendered)
    header_height = 10 + 15 * 2
    assert tuple(pixels[header_height + 20, 25]) == ROLE_COLORS["manipulated_object"]


def test_robot_tracking_experiment_is_strictly_segmentation_only() -> None:
    config_path = Path(__file__).resolve().parents[1] / "robot_tracking_config.json"
    config = robot_tracking_config(config_path)
    assert config["scope"] == {
        "entity": "whole_robot",
        "prompt_frame_index": 0,
        "process_every_frame": True,
        "shared_staging": "numeric JPEG, quality 100, chroma subsampling disabled",
        "generate_other_entities": False,
        "generate_scene_graph": False,
        "generate_depth_or_trajectory": False,
    }
    assert config["sam3"]["text_prompt"] == "robot"
    assert config["robotseg"]["category"] == "robot"


def test_robot_tracking_diagnostics_do_not_claim_accuracy() -> None:
    masks = []
    for offset in (0, 1, 2):
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 8 + offset:18 + offset] = True
        masks.append(mask)
    result = _diagnostics(masks)
    assert result["visible_frame_count"] == 3
    assert result["median_consecutive_mask_iou"] > 0.8
    assert "not accuracy metrics" in result["note"]
