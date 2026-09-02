# Minimal Unified Scene-Graph Pipeline

This directory is a self-contained, Docker-only pipeline for reference image sequences and generated MP4 videos. Its only semantic inputs are an ordered RGB sequence and a planning goal. It processes every frame and deliberately has no code path for ShareRobot planning steps, source masks, keyframes, aliases, confidence fields, or per-episode fixes.

## Stages

1. `prepare` normalizes every source frame to ordered RGB PNG without sampling.
2. `entities` asks the pinned local Qwen3.8-27B server for the five-role task specification, then tracks those fixed identities over all frames in deterministic five-frame box-or-null chunks.
3. `sam` runs both the detector-oriented text prompt and canonical visual name through native SAM3 video inference. The Qwen tracks select/reject SAM3 candidates per frame; masks and explicit status are written for every entity/frame.
4. `graph` asks Qwen to infer every frame from RGB, mask overlays, the goal, and role definitions.
5. `da3` reconstructs every frame with the vendored/patched DA3 streaming implementation.
6. `trajectory` robustly back-projects the manipulated-object mask through native DA3 depth, intrinsics, and camera-to-world poses.
7. `render` creates `visualization.mp4` and `contact_sheet.jpg`.
8. `validate` enforces the all-frame contract and public JSON invariants.

Each stage writes its final marker last. Stage fingerprints include the configuration and upstream artifacts, so a changed configuration or dependency reruns that stage and naturally invalidates downstream fingerprints. Directory-producing GPU stages build temporary output and swap it into place only after validation.

## Pinned software and models

`VERSIONS.json` records upstream revisions. Upstream licenses remain in each `third_party` tree. SAM3 and DA3 checkpoints are mounted read-only from `/datasets/unified_scene_graph_models`; Qwen is downloaded at the exact recorded Hugging Face revision using the pinned vLLM container. No checkpoint or secret is copied into an image.

The server runtime file defaults to `/datasets/unified_scene_graph_pipeline.runtime.env`. It must be mode `600` and contain `INFERENCE_API_KEY`, `HF_TOKEN`, and the three storage roots. `runtime.env.example` documents the fields and is safe to commit. A non-logging bearer-auth proxy protects the local endpoint; the vLLM process never receives the key as a command-line argument and is not published on a host port.

## Server use

```bash
chmod 600 /datasets/unified_scene_graph_pipeline.runtime.env
./run.sh build
./run.sh download-qwen
./run_experiment.sh manifests/pilot_11.json \
  /datasets/sharerobot_planning_selected /runs/pilot_v1
```

The experiment runner deliberately serializes GPU work on physical GPU 1: Qwen entity pass → stop Qwen → SAM3 → Qwen graph pass → stop Qwen → DA3. The normal Compose device reservation exposes only GPU 1. If single-GPU Qwen exits during startup, the runner may use the TP=2 override only after verifying that GPU 0 has at least 40 GB free; GPU 0 is otherwise untouched.

For one generated MP4, use the identical stage commands, changing only `prepare`:

```bash
./run.sh prepare --output /runs/generated_smoke --video /datasets/input.mp4 \
  --planning-goal "place the object in the container"
```

Then run `entities`, `sam`, `graph`, `da3 --scene`, `trajectory`, `render`, and `validate` on that output. No frame-count option is required.

## Cohorts and stopping rule

- `manifests/pilot_11.json` is the fixed one-per-dataset pilot.
- `select_validation.py` deterministically selects 100 scenes with all 11 datasets, all observed image resolutions, and forced task-family coverage. It uses only manifest goals and PNG headers.
- `review-index` writes JSON and HTML review indexes with the agreed failure taxonomy and visual checklist.

The scripts do not contain a full-1,000 run command. The 100-scene validation must be reviewed and summarized before any broader run is launched.

## Current pilot result

Configuration `unified_goal_roles_role_specific_v5` + `unified_split_state_action_v8` completed technical validation for all 11 pilot scenes. Manual semantic outcome: **1 fully successful, 4 partially successful, and 6 unsuccessful**. The split graph pass improves Bridge, EDAN, and UCSD, but regresses FMB and the clamp scene; RoboSet still has an incorrect target role. Consequently, `manifests/validation_100.json` is prepared but has not been launched. See `reports/pilot_v8_failure_summary.json` for the current per-scene assessment.

## Public output tree

Each scene contains `input.json`, `frames/`, `task_spec.json`, `masks/`, `tracks.json`, `scene_graph.json`, `da3/`, `trajectory_3d.json`, `visualization.mp4`, `contact_sheet.jpg`, `validation_report.json`, and `run_report.json`. Qwen request/response audits are kept separately for reproducibility and contain no API key.

## Whole-robot tracking comparison

`run_robot_tracking_experiment.sh` is a segmentation-only comparison between the official RobotSeg automatic video mode (`category="robot"`) and SAM3 native video propagation from the single text prompt `robot`. It processes every frame in `manifests/pilot_11.json` and deliberately skips Qwen, all non-robot entities, relations, actions, DA3, and trajectories.

The run writes per-method masks and diagnostics, plus a three-panel `visualization.mp4` and `contact_sheet.png` for every scene. The diagnostics describe temporal mask behavior without ground-truth robot masks; they are not segmentation-accuracy metrics.
