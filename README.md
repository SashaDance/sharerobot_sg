# Unified Scene-Graph Pipeline

The active implementation is [`unified_scene_graph_pipeline/`](unified_scene_graph_pipeline/).
It uses only ordered RGB frames and a planning goal, and it runs the same all-frame
pipeline for reference episodes and generated videos.

Historical implementations and the earlier evaluation/dashboard code are preserved
unchanged under [`legacy_code/`](legacy_code/). Data, papers, reports, checkpoints,
secrets, run outputs, and experiment artifacts are intentionally not part of source
control.

Each pipeline experiment is identified by the full source commit hash and stored under
`experiments/<commit-hash>/`. Every experiment contains the same eleven pilot
visualizations and a concise manual visual assessment.
