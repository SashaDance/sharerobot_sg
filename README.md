# Goal-Guided Robot Video Segmentation

The reusable Docker pipeline is in
[`unified_scene_graph_pipeline/`](unified_scene_graph_pipeline/). It segments
task-relevant entities in a robot-manipulation video using only ordered RGB
frames and a natural-language planning goal.

This branch intentionally excludes scene-graph evaluation experiments, reports,
manifests, generated outputs, depth reconstruction, and one-off server launch
scripts. See the pipeline README for installation and usage.
