# Goal-Guided Robot Video Segmentation

This is a Docker-only pipeline for segmenting task-relevant entities in a robot
manipulation video. Its only semantic inputs are a natural-language planning goal
and either an ordered RGB image sequence or an MP4. It processes every frame and
does not use dataset annotations, source masks, planning steps, keyframes, or
episode-specific rules.

## Method

The current configuration uses the same procedure for every video:

1. `prepare` converts the complete input to numerically ordered RGB PNG frames.
2. `entities` uses Qwen3.8-27B to identify the robot, manipulated object,
   initial support, target, and optional whole parent. It then obtains six
   uniformly spaced box-or-null grounding anchors per 30-frame window and repeats
   grounding after an all-frame entity-reflection pass.
3. `sam` runs SAM3 video text segmentation with the entity prompt and canonical
   name. Sparse Qwen anchors select the intended SAM3 track. SAM2.1 propagates
   from the same anchors and fills frames where the selected SAM3 track is absent.

The Qwen boxes are grounding cues rather than final masks. There are no
confidence fields in the public outputs.

## Requirements

- Linux with Docker Engine, Docker Compose v2, the NVIDIA Container Toolkit,
  and `curl` on the host.
- One CUDA GPU capable of serving the 55.6 GB BF16 Qwen model. Qwen and
  segmentation run sequentially, so they do not need to fit concurrently.
- A Hugging Face token that can download `Qwen/Qwen3.8-27B`.
- The SAM3 checkpoint `sam3.pt` obtained under its upstream access terms.

Pinned source revisions, model revisions, and checkpoint hashes are listed in
`VERSIONS.json`. SAM3 and SAM2 source and licenses are retained under
`third_party/`.

## Setup

Create a server-only runtime file outside the repository:

```bash
cp runtime.env.example /datasets/goal_guided_segmentation.runtime.env
chmod 600 /datasets/goal_guided_segmentation.runtime.env
```

Edit it and set `INFERENCE_API_KEY`, `HF_TOKEN`, storage roots, and the physical
GPU IDs. `INFERENCE_API_KEY` is a private bearer token of your choice used to
protect the local Qwen endpoint; it is not a paid external API key.

Build the core image and download the pinned models:

```bash
export SEGMENTATION_ENV=/datasets/goal_guided_segmentation.runtime.env
./run.sh build
./run.sh download-qwen
./run.sh download-sam2
```

Place SAM3 at `${MODEL_ROOT}/sam3.pt`. Startup verifies these hashes:

```text
sam3.pt                 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
sam2.1_hiera_large.pt   2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318
```

Do not put tokens or checkpoints into the repository or Docker image.

## Segment one video

Paths passed to the pipeline are container paths. `${DATASET_ROOT}` is mounted
read-only at `/datasets`, and `${RUN_ROOT}` is mounted at `/runs`.

```bash
export SEGMENTATION_ENV=/datasets/goal_guided_segmentation.runtime.env

./run.sh prepare \
  --output /runs/example \
  --video /datasets/input/example.mp4 \
  --planning-goal "place the red block in the metal bowl"

./run.sh start-qwen
./run.sh wait-qwen
./run.sh entities --output /runs/example
./run.sh stop-qwen

./run.sh sam --output /runs/example
```

For an image sequence, replace `--video` with `--images`. Filenames must contain
numeric frame indices. A JSON goal can be passed with `--planning-goal-json`; the
file must contain a `planning_goal` string.

## Batch mode

Create a local `manifests/run.json` file:

```json
{
  "episodes": [
    {"relative_path": "dataset_a/episode_0001"},
    {"relative_path": "dataset_b/episode_0042"}
  ]
}
```

Each source directory must contain `images/` and `planning_goal.json`. The batch
stages deliberately remain separate so Qwen can be stopped before SAM starts:

```bash
./run.sh batch --stage prepare \
  --manifest /manifests/run.json \
  --source-root /datasets/source \
  --output-root /runs/run

./run.sh start-qwen
./run.sh wait-qwen
./run.sh batch --stage entities \
  --manifest /manifests/run.json \
  --source-root /datasets/source \
  --output-root /runs/run
./run.sh stop-qwen

./run.sh batch --stage sam \
  --manifest /manifests/run.json \
  --source-root /datasets/source \
  --output-root /runs/run
```

Pass `--fail-fast` to stop a batch at its first failed scene. Without it, the
batch writes a report and continues.

## Outputs

Each output scene contains:

```text
input.json
frames/frame_000000.png ...
task_spec.json
qwen_entities_audit.json
tracks.json
masks/<entity_id>/frame_000000.png ...
run_report.json
.stages/*.json
```

Masks are single-channel PNGs with the same dimensions and frame correspondence
as the prepared RGB sequence. A missing entity or failed frame is represented by
an empty mask and an explicit status in `tracks.json`.

Stages are atomic and resumable. A stage is skipped only when its configuration
and upstream-artifact fingerprint matches the stored marker. Use `--overwrite`
to force a stage to run again.

## Scope and limitations

The five-role ontology is designed for manipulation tasks. Optional roles may be
null when they are not visually present. Low-resolution videos, severe
occlusion, tiny manipulated objects, and ambiguous planning goals remain common
failure modes. The saved Qwen audit is intended for diagnosing these cases; it
contains prompts and model responses but no API key.

This branch provides segmentation only. It does not generate relations, actions,
depth, trajectories, evaluation metrics, or visual-review dashboards.

## Tests

```bash
pytest -q tests
```

The tests cover frame preservation, schemas, sparse anchors, Qwen retries,
candidate prompts, and SAM3/SAM2 mask selection without requiring model weights.

## Third-party licensing

The upstream licenses for SAM3 and SAM2 are retained in their vendor trees.
Review `THIRD_PARTY.md` and the upstream model terms before redistribution. The
repository currently does not declare a license for the original pipeline glue
code; the maintainer should choose one before public release.
