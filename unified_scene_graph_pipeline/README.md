# Unified Scene-Graph Pipeline

This is the reproducible Docker implementation of the goal-guided robot-video
scene-graph pipeline. Its only semantic inputs are a natural-language planning
goal and either an ordered RGB sequence or an MP4. It processes every frame and
does not consume ShareRobot planning steps, source masks, keyframes, confidence
fields, or episode-specific corrections.

## Pipeline

The same stages are used for reference and generated videos:

1. `prepare` converts every input frame to numerically ordered RGB PNG files.
2. `entities` uses Qwen3.8-27B to identify the robot, manipulated object,
   initial support, target, and optional whole parent. It obtains six uniformly
   spaced box-or-null grounding anchors per 30-frame window and performs an
   all-frame entity-reflection pass.
3. `sam` runs native SAM3 video text segmentation. Sparse Qwen boxes select the
   intended SAM3 track, and SAM2.1 fills frames where that track is absent.
4. `graph` uses Qwen over the complete video to produce one state/action graph
   per frame and verify the temporal event sequence.
5. `da3` reconstructs every frame with the pinned Depth Anything 3 streaming
   implementation, saving depth, intrinsics, poses, and point clouds.
6. `trajectory` back-projects the manipulated-object masks through the DA3
   geometry to form its 3D trajectory.
7. `render` writes a review video and contact sheet with entity labels and
   readable graph edges.
8. `validate` enforces the all-frame artifact and JSON-schema contracts.

Stages write their marker last and use fingerprints of their inputs and
configuration. Rerunning the pipeline skips valid work and resumes at the first
missing or invalid stage. GPU directory outputs are built in temporary paths and
installed only after validation.

## Requirements

- Linux, Docker Engine, Docker Compose v2, NVIDIA Container Toolkit, and `curl`.
- One CUDA GPU with enough memory for the 55.6 GB BF16 Qwen model. Qwen,
  segmentation, and DA3 run sequentially.
- A Hugging Face token with access to `Qwen/Qwen3.8-27B`.
- `sam3.pt`, obtained under the upstream model terms.

Pinned repository/model revisions are recorded in `VERSIONS.json`; upstream
licenses are retained under `third_party/`. Checkpoints and secrets are never
copied into images or committed to Git.

## Setup

Create a server-only environment file outside the repository:

```bash
cp runtime.env.example /datasets/unified_scene_graph_pipeline.runtime.env
chmod 600 /datasets/unified_scene_graph_pipeline.runtime.env
```

Fill in `INFERENCE_API_KEY`, `HF_TOKEN`, and the storage roots. The inference key
is a private bearer token protecting the local Qwen endpoint, not a paid API
credential. GPU assignments can be configured with `QWEN_GPU_ID`,
`CORE_GPU_ID`, and `DA3_GPU_ID`; all default to physical GPU 1.

Build the images and download the pinned models:

```bash
export UNIFIED_SGG_ENV=/datasets/unified_scene_graph_pipeline.runtime.env
./run.sh build
./run.sh download-qwen
./run.sh download-sam2
```

Place SAM3 at `${MODEL_ROOT}/sam3.pt` and DA3 at
`${MODEL_ROOT}/da3/model.safetensors` with its configuration at
`${MODEL_ROOT}/da3/config.json`. Expected immutable revisions and checkpoint
hashes are listed in `VERSIONS.json` and the runtime configuration.

## Run one video

Paths passed to the commands are container paths. `${DATASET_ROOT}` is mounted
read-only at `/datasets`; `${RUN_ROOT}` is mounted read-write at `/runs`.

```bash
export UNIFIED_SGG_ENV=/datasets/unified_scene_graph_pipeline.runtime.env

./run.sh prepare --output /runs/example \
  --video /datasets/input/example.mp4 \
  --planning-goal "place the red block in the metal bowl"

./run.sh start-qwen
./run.sh wait-qwen
./run.sh entities --output /runs/example
./run.sh stop-qwen

./run.sh sam --output /runs/example

./run.sh start-qwen
./run.sh wait-qwen
./run.sh graph --output /runs/example
./run.sh stop-qwen

./run.sh da3 --scene /runs/example
./run.sh trajectory --output /runs/example
./run.sh render --output /runs/example
./run.sh validate --output /runs/example
```

For ordered images, replace `--video` with `--images`. A goal JSON can be passed
with `--planning-goal-json`; it must contain a `planning_goal` string.

## Batch mode

Copy `manifests/example.json` and list source-relative scene directories. Every
source scene must contain `images/` and `planning_goal.json`.

```bash
./run_pipeline.sh manifests/example.json \
  /datasets/source_dataset /runs/my_scene_graph_run
```

The generic runner serializes Qwen, SAM, graph generation, and DA3 so the large
models do not have to coexist on the GPU. It stops Qwen before each non-Qwen GPU
stage and generates a final review index. The underlying stage markers make the
command safe to rerun after interruption. In batch mode, one persistent DA3
container loads the checkpoint once and reuses the model for all pending scenes.

## Outputs

Each scene contains:

```text
input.json
frames/
task_spec.json
masks/
tracks.json
scene_graph.json
da3/
trajectory_3d.json
visualization.mp4
contact_sheet.jpg
validation_report.json
run_report.json
```

`scene_graph.json` contains one record per input frame. State predicates are
`on`, `inside`, `holding`, `touching`, `part_of`, `attached_to`, `open`, and
`closed`. Actions are `reach_for`, `grab`, `lift`, `move`, `place`, `release`,
`push`, `pour`, `open`, `close`, `insert`, and `stack`.

DA3 normally saves its native confidence-filtered point cloud. For a
low-confidence clip where native downsampling would produce zero vertices, the
wrapper regenerates the cloud from the already validated native depth,
intrinsics, and poses while retaining every point that passes DA3's confidence
rule. The use of this global fallback is recorded in the DA3 stage metadata.

## Release scope

This branch contains only the production segmentation, scene-graph, DA3,
trajectory, visualization, and validation paths. Historical experiment outputs,
dataset curation scripts, RobotSeg, InterRVOS, and server-specific launch scripts
are deliberately excluded.
