# 100-scene validation audit

Run: `48d934e85ddd8b93a570f647f7f5c103a0dbdd01`

The deterministic validation run completed all pipeline stages for all 100
scenes. JSON schemas, frame correspondence, graph references, depth, pose, and
render artifacts passed the automated validator in every scene.

## Quantitative audit

- 100/100 scenes completed entity extraction, SAM tracking, graph inference,
  DA3, trajectory generation, rendering, and validation.
- The automated review marked two scenes with a trajectory failure:
  `60_plex_robosuite/episode_75` and `62_robo_set/episode_5990`.
- Nine scenes contain at least one null trajectory point; there are 72 null
  points in total. These are caused by missing manipulated-object masks rather
  than missing or non-finite DA3 output.
- Required robot and manipulated-object tracks exist in all 100 scenes. Mean
  visible-frame coverage is 0.981 for the robot and 0.976 for the manipulated
  object.
- SAM2 missing-frame repair is used in 18 robot tracks (87 frames) and 37
  manipulated-object tracks (858 frames).
- Initial support is present in 95 scenes, target in 64, and whole parent in one
  open-oven scene. Whole parent therefore remains necessary for the active task
  distribution.
- Graph generation used one schema retry in 38 scenes and recovered all 38;
  there are no invalid final graph files.
- Eleven scenes assign exactly the same canonical entity to initial support and
  target. This is a systematic role co-reference issue and should be addressed
  globally rather than with aliases or episode-specific rules.

## Visual findings

- DLR Pour episode 0, UCSD episodes 1026 and 964, and Bridge episode 9130 show
  that the current pipeline can produce coherent masks and state transitions.
- VIOLA episode 112 exposes a graph-only error: the correct bowl-on-plate final
  relation is accompanied by an unsupported target-on-initial-support relation.
- PLEX episode 75 loses the manipulated object through a long occlusion. The
  inferred placement is plausible, but 17 trajectory points are invalid.
- ASU episode 83 is sensitive to entity grounding: the 100-scene run names the
  task object as an orange box, while a later nominally graph-only rerun selects
  an unrelated green bottle. DLR Clamp episode 17 remains difficult because of
  repeated instances.
- DLR EDAN episode 5 and RoboSet episode 5990 do not provide enough visible
  evidence for the requested task. They should be reported as data-eligibility
  failures, not used to justify scene-specific tracker changes.

## Decisions

1. Keep DA3 unchanged. Its outputs are complete and internally consistent; the
   observed 3D trajectory failures originate in segmentation coverage.
2. Keep SAM3 as the primary tracker with SAM2 used only for missing-frame
   repair. A global SAM2 selection experiment regressed otherwise good robot
   and target tracks.
3. Continue graph-evidence ablations because some relation errors occur despite
   usable masks. Freeze and reuse entity and mask artifacts for these ablations;
   rerunning Qwen entity extraction confounds a graph-only comparison even at
   temperature zero. Evaluate changes on the fixed ten-dataset pilot before any
   further 100-scene run.
4. Add a global role co-reference policy and a separate insufficient-visual-
   evidence status in a later iteration. Neither should silently rewrite an
   episode's result.

This audit is structural and diagnostic; it is not a claim that all 100 scenes
are semantically correct. The fixed pilot is used for controlled visual
comparison, while broader semantic accuracy requires additional human review.
