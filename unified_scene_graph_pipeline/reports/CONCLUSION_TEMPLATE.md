# Experiment conclusion

Experiment: `<short description>`  
Commit: `<full source commit hash>`  
Outcome count: N fully successful, M partially successful, and K unsuccessful among V valid scenes; E scenes excluded because the planning goal does not match the video.

N + M + K must equal V, while V + E must equal the number of manually reviewed scenes. Report technical
validation separately; a scene that completes without errors is not automatically
semantically successful.

Use these categories consistently:

- `fully successful`: task-critical entities, masks, state change, and actions are semantically usable.
- `partially successful`: the task remains interpretable, but at least one material component is missing or wrong.
- `unsuccessful`: the output does not provide a reliable task interpretation.
- `excluded_invalid_planning_goal`: the video and planning goal are visibly inconsistent. This is an input-data error, not a pipeline failure, and is excluded from all success/failure denominators.

Follow the metric with the automated result, one short visual conclusion per scene,
the overall conclusion, and the recommended next global revision or stopping reason.
