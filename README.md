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

Every experiment conclusion must include this exact summary near the top:
`Outcome count: N fully successful, M partially successful, and K unsuccessful.`
The three values must sum to the reviewed scene count. “Fully successful” means
the task-critical entities, masks, state change, and actions are semantically
usable; “partially successful” means the task is still interpretable but at least
one material component is missing or wrong; “unsuccessful” means the output does
not provide a reliable task interpretation. Technical completion without crashes
is reported separately and does not determine this semantic outcome count.
