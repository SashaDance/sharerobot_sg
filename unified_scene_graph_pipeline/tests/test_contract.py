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
    validate_action_document,
    validate_entity_document,
    validate_state_document,
)
from render import ROLE_COLORS, _draw_frame  # noqa: E402


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
            "robot": {"canonical_name": "gray robot arm", "sam_prompt": "gray robot arm"},
            "manipulated_object": {"canonical_name": "blue cube", "sam_prompt": "small blue cube"},
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
    try:
        ensure_no_confidence({"confidence": 0.9})
        raise AssertionError("confidence accepted")
    except ValueError:
        pass


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
