# jianying-ai-image-aligner agent rules

## Current valid route

Only use the direct draft-write route:

1. `src/direct_draft_broll_writer.py`
2. `src/pipeline_contract_check.py`
3. `src/create_test_visual_slot_package.py`
4. `run_direct_draft_write.ps1`
5. `run_pipeline_contract_check.ps1`
6. `run_create_test_visual_slot_package.ps1`
7. `run_negative_tests.ps1`
8. `run_preproduction_check.ps1`

The tool must be called with the current project's:

- `DraftDir`
- `BrollMd`
- `ImageDir`
- `VisualSlotPlan`

Do not rely on `agent_inputs.json` or historical S16 paths.

## Confirmation gate

Only run draft writing after the user has explicitly confirmed the AI image batch is usable.

This tool is stage 4 of the chain:

1. A-Roll QC passed.
2. B-roll design confirmed by user.
3. AI image quality confirmed by user.
4. Draft alignment/write and report.

Do not auto-run alignment immediately after image generation.

## Generic contract

- AI image count is derived from the B-roll design, normalized image directory, and `visual_slot_plan.json`.
- AI image filenames must contain `_AI_<number>_`.
- The B-roll table IDs, AI static-list IDs, normalized image filenames, and visual slot image IDs must match exactly.
- The aligner must not author real B-roll creative decisions. Production consumes the B-roll agent output. Test-package generation requires `-ReferenceBroll` and only preserves the real design structure while using current-draft captions for distributed timing.
- If `visual_slot_plan.json` carries confidence values below threshold, execution must stop.
- Each written image clip uses `target_end_us - target_start_us`; fixed `1.3s` duration is forbidden.
- The target start/end/duration must match the slot exactly. Source duration may tolerate only a `1us` source-only normalization drift after Jianying reopens the draft.
- The written track name is `AI_BROLL`.
- Slot intervals must not overlap on the single `AI_BROLL` track.
- Slot intervals must not cross video target gaps or exceed their `container_video_segment_ids`.

## Version Boundary

Current v0.2:

- reads `visual_slot_plan.json`
- aligns each image from `target_start_us` to `target_end_us`
- uses slot interval duration
- writes the independent `AI_BROLL` image track
- performs post-write actual image audit before GUI QC

This is the only supported implementation route. The old v0.1 code path was removed from the tool, including:

- fixed-duration `1.3s` planners
- screenshot/UI builder route
- drag-and-drop execution route
- old overlay/SRT-style align route
- S16-specific plan builders

## Runtime and workspace boundaries

Runtime output is external:

`D:\auto_clip_runtime\image_aligner`

IDEA Codex:

- Only edit `src`, run scripts, and docs.
- Do not scan `D:\auto_clip_runtime`.
- Do not scan `vendor` unless explicitly requested.
- Do not read historical runtime JSON by default.
- Do not execute real UI drag operations.
- Do not write real Jianying drafts unless the user explicitly confirms the write stage.
- Do not batch-generate images.

Desktop Codex:

- Handles runtime migration.
- Handles real dry-run / execute workflows.
- Handles long tasks and batch tasks.
- Must back up before writing real drafts.
- For end-to-end validation, use `run_create_test_visual_slot_package.ps1 -ReferenceBroll <real-design.md>` to copy 10 images from the user's AI image directory into an isolated test package. It must draw distributed caption slots from the current draft and preserve the real B-roll design structure, then run preflight and a single confirmed write against the specified draft.
- Before trusting a new draft/design class, run `run_negative_tests.ps1` against the latest valid package. The rollback case must use a disposable draft clone, not the user's active draft.
- Before a real `-ConfirmWrite` on production inputs, run `run_preproduction_check.ps1`. It must pass preflight-only validation and negative sweep without writing the active draft. Run post-write actual contract check after the confirmed write.

## Removed routes

Do not recreate the old UI, screenshot, SRT, overlay-video, drag, or fixed `1.3s` routes. They were removed rather than kept as compatibility wrappers. If a caller needs image alignment, route it through `run_direct_draft_write.ps1`, `run_pipeline_contract_check.ps1`, `run_preproduction_check.ps1`, or `run_create_test_visual_slot_package.ps1`.

## Local config

`agent_inputs.example.json` is documentation only. Keep real local paths out of OSS export; `agent_inputs.json` is ignored and must not be used as an automatic fallback.
