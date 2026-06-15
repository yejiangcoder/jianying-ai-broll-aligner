# Jianying AI Image Aligner

用途：给 agent 使用，把 B-ROLL 设计稿里的 AI 静态图直接写进剪映 10.7 prepared 草稿，生成独立图片片段轨道 `AI_BROLL`。

## 当前锁定路线

只走草稿级自动化：

1. 用 `jy-draftc` 调用剪映安装目录里的 `videoeditor.dll` 解密当前 prepared 草稿主体 `draft_content.json`。
2. 从明文 `draft_content` 的最终字幕 text track 读取 `subtitle_text / start_sec / end_sec`。
3. 解析 B-ROLL 设计稿里的 `AI静态图`：编号、台词落点、画面设计。
4. 用 `台词落点` 语义匹配最终字幕文本，生成 `broll_exec_plan`。
5. 新建或复用 `AI_BROLL` 图片轨道。
6. 把 AI 静态图按字幕起点写入草稿，数量以 B-ROLL 设计稿和规范图片目录为准，每张固定 `1.3s`。
7. 回加密并覆盖当前工程草稿主体。
8. 重新解密当前草稿自检。

## 禁止路线

- 不做 UI 拖拽。
- 不截图识别字幕块。
- 不导出 SRT。
- 不做轨道坐标校准。
- 不生成 MP4 覆盖视频。
- 不处理现实素材。
- 不创建辅助草稿复制轨道。

## 输入契约

配置文件：

`D:\video tools\jianying-ai-image-aligner\agent_inputs.json`

关键字段：

- `draft_dir`：当前剪映工程目录。
- `broll_md`：B-ROLL 设计稿。
- `ai_image_dir`：AI 静态图目录。
- `duration_sec`：固定 `1.3`。

AI 图片只识别规范命名：

```text
项目名_AI_02_画面短名.png
项目名_AI_100_画面短名.png
```

文件名必须包含 `_AI_编号_`。目录里的 `ChatGPT Image ...png` 这类原始文件不会进入施工计划。

## B-ROLL 设计稿要求

每张 `AI静态图` 必须至少有：

- 编号，例如 `02`
- `B-roll类型 = AI静态图`
- `台词落点`
- `对齐台词起句`；当 `台词落点` 是画面意图、概括句或改写句，而不是字幕原句时必须写。
- `画面设计` 或 AI 清单里的 `画面方向`
- AI 静态图清单里的 `【编号】`

`台词落点` 就是自动对齐的 `target_text`。  
`台词落点` 和 `对齐台词起句` 会用于匹配最终字幕。匹配置信度低于阈值时工具会停止，不会硬写入。

## 命令

直接写当前剪映草稿：

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "D:\video tools\jianying-ai-image-aligner\src\direct_draft_broll_writer.py"
```

三位一体自检：

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "D:\video tools\jianying-ai-image-aligner\src\pipeline_contract_check.py"
```

## 输出

每次写入会生成：

```text
D:\auto_clip_runtime\image_aligner\runs\direct_write_时间戳\
```

包含：

- `draft_content.dec.json`
- `draft_content.modified.json`
- `draft_content.encrypted.json`
- `broll_exec_plan.csv`
- `verify_after_write.dec.json`，自检时生成

## 成功标准

- `AI_BROLL` 轨道存在。
- B-ROLL 设计稿、AI 静态图清单、规范图片目录的编号完全一致。
- 规范命名的 AI 静态图全部写入。
- 每张图片时长都是 `1.3s`。
- 图片路径指向项目原始 AI 静态图目录。
- AI 轨道在滤镜轨上方、字幕轨下方。
- 同一轨道无重叠。

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

## Open Source Export Notice

This repository contains source code only.

Runtime artifacts, Jianying drafts, media files, generated images, local configs, cookies, and API keys are not included.

Configure local runtime paths with environment variables or example config files.
