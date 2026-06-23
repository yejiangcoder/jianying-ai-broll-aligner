# Jianying AI Image Aligner

用途：把已 QC 通过的 AI 静态图按 `visual_slot_plan.json` 写进当前剪映草稿的独立 `AI_BROLL` 图片轨道。

## Current Route

只使用直接草稿写入路线：

1. `src/direct_draft_broll_writer.py`
2. `src/pipeline_contract_check.py`
3. `src/create_test_visual_slot_package.py`
4. `run_direct_draft_write.ps1`
5. `run_pipeline_contract_check.ps1`
6. `run_create_test_visual_slot_package.ps1`
7. `run_negative_tests.ps1`
8. `run_preproduction_check.ps1`

旧 UI / 拖拽 / 固定 1.3s / overlay 对齐路线已经从代码库移除，不再提供兼容入口。后续图片对齐只能走 v0.2 direct draft-write route。

必须显式传入当前项目的：

- `DraftDir`
- `BrollMd`
- `ImageDir`
- `VisualSlotPlan`

`jy-draftc` 从 `JY_DRAFTC_EXE` / `JY_DRAFTC` 读取，或通过 `-JyDraftc` / `--jy-draftc` 显式传入。仓库不再提供硬编码 vendor 默认值。

## Pipeline Stage

图片对齐是 `visual_slot_plan` 生成/校验之后的写图阶段：

```text
A-Roll 通过 QC -> B-Roll 设计稿通过 QC -> AI 批量跑图通过 QC -> visual_slot_plan 生成/校验 -> 图片对齐写入 -> GUI QC
```

写入前仍有确认门槛：没有 `-ConfirmWrite` / `--confirm-write` 时，只生成 preflight 确认单，不写草稿。

## visual_slot_plan

本阶段不再猜字幕轨，也不重新做语义对齐。时间轴来源只能是上游输出的 `visual_slot_plan.json`。

标准上游生成器：

```text
D:\video tools\broll-designs\scripts\create_visual_slot_plan_from_broll.ps1
```

该生成器消费已 QC 的 B-Roll 设计稿、`broll_handoff_manifest.json`、人工精剪最终字幕时间轴和已 QC 图片目录。只有 `visual_slot_plan.json.ready_for_image_alignment=true` 才允许进入本工具。

```json
{
  "slots": [
    {
      "slot_id": "broll_001",
      "text": "对应台词",
      "target_start_us": 1230000,
      "target_end_us": 3450000,
      "source_start_us": 900000,
      "source_end_us": 3120000,
      "container_video_segment_ids": ["..."],
      "image_path": "D:\\path\\to\\image_AI_01_scene.png"
    }
  ]
}
```

写入规则：

- 图片 `target_start` 使用 `target_start_us`。
- 图片 `target_duration` 使用 `target_end_us - target_start_us`。
- 禁止固定 `1.3s` 默认。
- slot 不能跨 video target gap。
- slot 不能超过 `container_video_segment_ids` 对应的实际 video 区间。

## Contract

- AI 图片文件名必须包含 `_AI_<number>_`。
- B-roll 表格 ID、AI 静态图清单 ID、规范图片目录 ID、visual slot 图片 ID 必须完全一致。
- 图片对齐工具不主观创作 B-Roll 设计；正式流程消费 B-Roll agent 产物，测试包也必须传真实设计稿作为结构参考。
- 如果 slot plan 携带 confidence，低于阈值会停止执行。
- 写入轨道名固定为 `AI_BROLL`。
- 写入前会移除旧 `AI_BROLL`，再写入新的 slot 集合。

## Commands

Create an isolated 10-image test package from a real draft and AI image directory：

```powershell
.\run_create_test_visual_slot_package.ps1 `
  -DraftDir "D:\JianyingPro Drafts\CURRENT_PROJECT" `
  -SourceImageDir "D:\video\...\AI静态图目录" `
  -ReferenceBroll "D:\path\to\真实B-roll设计.md" `
  -Count 10
```

The generated package contains:

- `test_ai_images\`
- `broll_test_design.md`
- `visual_slot_plan.json`
- `test_package_manifest.json`

Preflight：

```powershell
.\run_direct_draft_write.ps1 `
  -DraftDir "D:\JianyingPro Drafts\CURRENT_PROJECT" `
  -BrollMd "D:\path\to\broll.md" `
  -ImageDir "D:\path\to\images" `
  -VisualSlotPlan "D:\path\to\visual_slot_plan.json"
```

Write after AI image QC：

```powershell
.\run_direct_draft_write.ps1 `
  -DraftDir "D:\JianyingPro Drafts\CURRENT_PROJECT" `
  -BrollMd "D:\path\to\broll.md" `
  -ImageDir "D:\path\to\images" `
  -VisualSlotPlan "D:\path\to\visual_slot_plan.json" `
  -ConfirmWrite
```

Post-write contract check：

```powershell
.\run_pipeline_contract_check.ps1 `
  -DraftDir "D:\JianyingPro Drafts\CURRENT_PROJECT" `
  -BrollMd "D:\path\to\broll.md" `
  -ImageDir "D:\path\to\images" `
  -VisualSlotPlan "D:\path\to\visual_slot_plan.json"
```

Negative contract sweep：

```powershell
.\run_negative_tests.ps1 `
  -BrollMd "D:\path\to\broll.md" `
  -ImageDir "D:\path\to\images" `
  -VisualSlotPlan "D:\path\to\visual_slot_plan.json"
```

This sweep must pass before treating the tool as production-ready for a new draft class. It uses disposable mutated packages and a disposable draft clone for rollback verification.

Preproduction check：

```powershell
.\run_preproduction_check.ps1 `
  -BrollMd "D:\path\to\broll.md" `
  -ImageDir "D:\path\to\images" `
  -VisualSlotPlan "D:\path\to\visual_slot_plan.json"
```

This is the final gate before a real `-ConfirmWrite`: it runs preflight-only input validation and the negative sweep. It does not write the active draft. Post-write actual contract check runs after `-ConfirmWrite`.

## Audit Gate

Post-write audit must pass before GUI QC:

- `image_slot_count == written_image_segment_count`
- 每个 image segment start/end/duration 精确匹配 slot
- 每个 image segment source duration 匹配 slot duration，允许剪映重新打开后的 `1us` source-only 量化误差
- image 不超过 containing video segment
- image 不跨 video target gap
- 无旧 `AI_BROLL` residue
- 无 image after final video end
- `only_specified_draft_written = true`
- root/timeline mirror consistent
- `post_write_actual_image_audit_gate_passed = true`

## Negative Gates

- missing image blocks
- extra image blocks
- B-roll table IDs and AI static-list IDs mismatch blocks
- visual slot image IDs and image directory mismatch blocks
- slot overlap blocks
- slot outside containing video segment blocks
- slot crossing video target gap blocks
- missing current draft binding blocks
- current draft state with A-Roll QC not passed blocks
- forced post-write audit failure restores the disposable draft clone hash

## Runtime

运行输出在外部 runtime：

```text
D:\auto_clip_runtime\image_aligner\runs
```

`agent_inputs.example.json` 只作为文档示例。真实本地路径请放在命令参数或本机私有配置中，`agent_inputs.json` 已加入 `.gitignore`。
