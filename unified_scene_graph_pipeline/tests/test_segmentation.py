from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))

from cli import _config, parser  # noqa: E402
from common import ensure_no_confidence, image_paths  # noqa: E402
from prepare import prepare  # noqa: E402
from qwen import (  # noqa: E402
    QwenClient,
    contextual_chunks,
    entity_reflection_prompt,
    sparse_anchor_indices,
    tracking_overlay_frame,
    validate_entity_document,
    validate_tracking_document,
)
from sam_stage import (  # noqa: E402
    _candidate_text_prompts,
    _prefer_primary_masks,
    _qwen_box_chunks,
    _select_hybrid_masks,
)


def test_public_cli_contains_only_segmentation_stages() -> None:
    choices = parser()._subparsers._group_actions[0].choices
    assert set(choices) == {"prepare", "entities", "sam", "batch"}
    batch_actions = choices["batch"]._actions
    stage = next(action for action in batch_actions if action.dest == "stage")
    assert set(stage.choices) == {"prepare", "entities", "sam"}


def test_release_configuration_loads() -> None:
    config = _config(str(Path(__file__).resolve().parents[1] / "config.json"))
    assert config["segmenter"]["backend"] == "sam3_sam2"
    assert config["qwen"]["box_anchor_count_per_window"] == 6
    assert "da3" not in config and "render" not in config and "ontology" not in config


def test_prepare_preserves_all_arbitrary_length_rgba_frames(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(7):
        Image.new("RGBA", (23, 17), (index, 20, 30, 120)).save(
            source / f"frame_{index}.png"
        )
    goal = tmp_path / "goal.json"
    goal.write_text(json.dumps({"planning_goal": "move the blue cube"}))
    output = tmp_path / "output"

    result = prepare(output, source, None, goal, None)

    assert result["frame_count"] == 7
    assert [path.name for path in image_paths(output / "frames")] == [
        f"frame_{index:06d}.png" for index in range(7)
    ]
    assert all(Image.open(path).mode == "RGB" for path in image_paths(output / "frames"))


def test_entity_and_tracking_schemas_reject_invalid_values() -> None:
    entities = {
        "roles": {
            "robot": {"canonical_name": "gray robot arm", "sam_prompt": "gray robot arm"},
            "manipulated_object": {"canonical_name": "blue cube", "sam_prompt": "small blue cube"},
            "initial_support": None,
            "target": None,
            "whole_parent": None,
        },
        "task_actions": ["move"],
    }
    validated = validate_entity_document(entities)
    assert validated["roles"]["robot"]["sam_prompt"] == "robot arm"

    tracking = {
        "tracks": {
            "robot": [[0, [100, 50, 900, 950]], [1, None]],
            "manipulated_object": [[0, [300, 300, 450, 500]], [1, [320, 300, 470, 500]]],
            "initial_support": None,
            "target": None,
            "whole_parent": None,
        }
    }
    normalized = validate_tracking_document(tracking, [0, 1], validated["roles"])
    assert normalized["robot"][1]["bbox_xyxy_1000"] is None

    tracking["tracks"]["manipulated_object"][1][1] = [450, 300, 300, 500]
    try:
        validate_tracking_document(tracking, [0, 1], validated["roles"])
    except ValueError:
        pass
    else:
        raise AssertionError("invalid grounding box accepted")

    try:
        ensure_no_confidence({"confidence": 0.9})
    except ValueError:
        pass
    else:
        raise AssertionError("confidence field accepted")


def test_context_chunks_never_exceed_model_image_limit() -> None:
    chunks = list(contextual_chunks(list(range(117)), 30))
    assert [index for current, _ in chunks for index in current] == list(range(117))
    assert all(len(current) + (previous is not None) <= 30 for current, previous in chunks)


def test_sparse_anchors_cover_every_thirty_frame_window() -> None:
    assert sparse_anchor_indices(list(range(30)), 6, 30) == [0, 6, 12, 18, 24, 29]
    assert sparse_anchor_indices(list(range(67)), 6, 30) == [
        0, 6, 12, 18, 24, 29,
        30, 36, 42, 48, 54, 59,
        60, 62, 63, 64, 65, 66,
    ]


def test_entity_reflection_overlay_uses_all_roles(tmp_path: Path) -> None:
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
    assert tracking_overlay_frame(frame, roles, grounding, 0).startswith(b"\x89PNG")
    prompt = entity_reflection_prompt(
        "move the blue cube", {"roles": roles, "task_actions": ["move"]}, [0]
    )
    assert "complete visible trajectory" in prompt


def test_role_specific_candidate_prompts_are_deduplicated() -> None:
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


def test_qwen_box_chunks_preserve_null_gaps() -> None:
    rows = [
        {"frame_index": index, "bbox_xyxy_1000": None if index in {1, 5} else [1, 2, 3, 4]}
        for index in range(7)
    ]
    chunks = _qwen_box_chunks(rows, frame_count=7, chunk_size=3)
    assert [(start, end) for start, end, _ in chunks] == [(0, 3), (3, 6), (6, 7)]
    assert [[row["frame_index"] for row in grounded] for _, _, grounded in chunks] == [
        [0, 2], [3, 4], [6]
    ]


def test_sam2_fills_only_missing_sam3_frames() -> None:
    primary = {0: np.asarray([[True, False]]), 2: np.asarray([[False, True]])}
    fallback = {0: np.asarray([[False, True]]), 1: np.asarray([[True, True]])}
    merged, fallback_frames = _prefer_primary_masks(primary, fallback)
    assert np.array_equal(merged[0], primary[0])
    assert np.array_equal(merged[1], fallback[1])
    assert np.array_equal(merged[2], primary[2])
    assert fallback_frames == [1]


def test_optional_hybrid_selector_prefers_box_aligned_source() -> None:
    sam3 = {
        index: np.asarray([[True, True, False, False, True, True]] * 2)
        for index in range(2)
    }
    sam2 = {
        index: np.asarray([[True, True, False, False, False, False]] * 2)
        for index in range(2)
    }
    grounding = [
        {"frame_index": index, "bbox_xyxy_1000": [0, 0, 334, 1000]}
        for index in range(2)
    ]
    masks, selection = _select_hybrid_masks(sam3, sam2, grounding, 1000)
    assert selection["selected_source"] == "sam2"
    assert np.array_equal(masks[0], sam2[0])


def test_qwen_retry_supplies_invalid_response_to_repair_prompt(monkeypatch) -> None:
    class Response:
        ok = True
        status_code = 200

        def __init__(self, content: str) -> None:
            self.content = content

        def json(self) -> dict:
            return {"choices": [{"message": {"content": self.content}}], "usage": {}}

    payloads = []
    responses = iter([Response('{"value":0}'), Response('{"value":1}')])

    def post(_url, headers, json, timeout):
        payloads.append(json)
        return next(responses)

    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    monkeypatch.setattr("qwen.requests.post", post)
    client = QwenClient({
        "qwen": {
            "base_url": "http://qwen/v1",
            "model": "test-model",
            "retries": 1,
            "temperature": 0,
        }
    })

    def validate(value: dict) -> dict:
        if value["value"] != 1:
            raise ValueError("value must be one")
        return value

    result, attempts = client.request(
        [{"type": "text", "text": "Return value"}], 100, validate
    )
    assert result == {"value": 1}
    assert len(attempts) == 2
    assert payloads[1]["messages"][1]["content"] == '{"value":0}'
    assert "value must be one" in payloads[1]["messages"][2]["content"]
