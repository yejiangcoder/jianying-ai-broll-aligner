# jianying-ai-image-aligner agent rules

## Current valid route

Only use the direct draft-write route:

1. `src/direct_draft_broll_writer.py`
2. `src/pipeline_contract_check.py`
3. `run_direct_draft_write.ps1`
4. `run_pipeline_contract_check.ps1`

The tool must be called with the current project's:

- `DraftDir`
- `BrollMd`
- `ImageDir`

Do not rely on the default S16-1 paths except when explicitly working on that project.

## Confirmation gate

Only run draft writing after the user has explicitly confirmed the AI image batch is usable.

This tool is stage 3 of the chain:

1. B-roll design confirmed by user.
2. AI image quality confirmed by user.
3. Draft alignment/write and report.

Do not auto-run alignment immediately after image generation.

## Generic contract

- AI image count is not fixed. It is derived from the B-roll design and normalized image directory.
- AI image filenames must contain `_AI_<number>_`.
- The B-roll table IDs, AI static-list IDs, and normalized image filenames must match exactly.
- If `台词落点` is not an exact subtitle phrase, the B-roll design must include `对齐台词起句`.
- Low-confidence subtitle matches must stop execution.
- Each written image clip is fixed at 1.3 seconds.
- The written track name is `AI_BROLL`.

## Version Boundary

Current v0.1:

- fixed `1.3s` image duration
- aligns each image to the matched subtitle start
- writes the independent `AI_BROLL` image track

Planned v0.2:

- reads `visual_slot_plan.json`
- aligns each image from `start_us` to `end_us`
- uses `duration_us` from the slot interval
- no fixed `1.3s` default
- still writes the independent `AI_BROLL` image track

v0.2 is not implemented yet. Desktop Codex must not assume `visual_slot_plan` execution works until IDEA Codex implements it.

`agent_inputs.json` may contain local project paths.
For portable usage, create `agent_inputs.example.json` and keep local paths out of OSS export.

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

## Forbidden routes

Do not run old UI, screenshot, SRT, overlay-video, or drag scripts directly, including:

- `align_ai_images.py`
- `broll_plan_pipeline.py`
- `jianying_ai_image_ui_builder.py`
- `jianying_drag_1p3s_builder.py`
- `make_*plan*.py`
- `set_jianying_image_default_duration.py`

These files remain only as historical reference. The PowerShell compatibility wrappers route away from them.
