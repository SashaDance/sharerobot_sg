# Goal-Guided Robot Video Scene Graphs

The reusable Docker pipeline is in
[`unified_scene_graph_pipeline/`](unified_scene_graph_pipeline/). Given ordered
RGB frames or an MP4 and a natural-language planning goal, it produces
task-relevant entity tracks and masks, frame-level state/action graphs, DA3
geometry, a manipulated-object 3D trajectory, and review visualizations.

This release branch intentionally excludes historical experiment artifacts,
dataset-selection utilities, one-off server launch scripts, and unrelated
tracking baselines. Model checkpoints, datasets, secrets, and generated outputs
are not stored in Git.

See the pipeline README for setup, single-video use, batch execution, output
schemas, and pinned third-party revisions.
