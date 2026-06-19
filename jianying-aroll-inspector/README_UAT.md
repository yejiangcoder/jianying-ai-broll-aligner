# A-Roll 工具使用方式

这个工具的 V21 入口是 `run_aroll_v21_operator.ps1`。真实草稿写回只能在显式授权、显式 `DraftDir`、显式 `RunDir` 的一次性 UAT 流程里执行。

## 日常用法

对 Codex 说：

```text
调用 A-ROLL 剪辑工具，开始工作。
```

Codex 会使用 V21 operator，先做 dry-run / pre-audit。没有明确授权时不会写回草稿。

V20 legacy wrappers and Phase 4/5 scripts have been removed from the active source tree and are not entrypoints.

## 指定草稿

对 Codex 说：

```text
用你的已准备草稿跑 A-Roll。
```

内部等价于：

```powershell
.\run_aroll_v21_operator.ps1 -DraftDir "YOUR_DISPOSABLE_DRAFT_DIR" -RunDir "YOUR_RUN_DIR" -Mode dry-run
```

## 只做预检

对 Codex 说：

```text
只做预检，先别写回。
```

内部等价于：

```powershell
.\run_aroll_v21_operator.ps1 -DraftDir "YOUR_DISPOSABLE_DRAFT_DIR" -RunDir "YOUR_RUN_DIR" -Mode dry-run
```

## 剪映开着怎么办

默认规则：如果剪映主进程正在运行，工具拒绝写回，避免草稿被剪映覆盖。

你可以先在剪映里保存，然后关闭剪映。

如果你明确授权，可以对 Codex 说：

```text
帮我关闭剪映再跑。
```

先手动关闭剪映，再让 Codex 在显式授权的 V21 UAT turn 中运行。

## 默认支持

- 1.0x 主视频
- 1.2x constant speed 主视频
- <= 1.25x constant speed 主视频
- 主视频 segment/material 上的音量、人声增强、降噪、figure / beauty refs 克隆保留
- 写后检查 speed refs、volume fields、audio enhancement、attached refs 是否保留
- Semantic gate 使用“本地候选发现 + DeepSeek 语义仲裁 + 保守 source keep 修复”。DeepSeek 只裁决语义，不生成 EDL，不写草稿。

## 当前不支持

- 曲线变速
- 倒放
- mixed speed
- 独立 audio track
- 全局 audio / filter track
- 无法识别或无法克隆的 attached refs

## Profile 模板

公开 release 默认只提供：

```text
profiles\production.json
profiles\default_profile.json
```

这些文件里的 `EDIT_ME_DRAFT_ROOT` / `EDIT_ME_DRAFT_NAME` 是模板占位符。V21 真实运行必须显式传入 `-DraftDir` 和 `-RunDir`；否则工具会阻塞。

核心默认值：

```text
default_draft_name = YOUR_DRAFT_NAME
allow_constant_speed = true
max_allowed_speed = 1.25
runtime_mode = production
run_cleanup_before = true
run_cleanup_after = true
keep_debug_dec_json = false
keep_audio_pcm = false
auto_close_jianying = false
```

## 你需要验收什么

工具完成后重点检查：

1. 是否有重复句
2. 是否切字 / 断尾音
3. 字幕是否不长、不碎、不飞轨
4. 1.2x 变速是否正常
5. 声音增强是否仍在
6. 气口是否可接受

重点报告：

```text
run_summary.json
prewrite_report.json
writeback_report.json
postwrite_report.json
validator_report.json
semantic_request_payloads.json
semantic_adjudication_report.json
deepseek_decisions.json
```

## 附录：V21 入口

```powershell
cd "<suite-root>\jianying-aroll-inspector"
.\run_aroll_v21_operator.ps1 -DraftDir "EDIT_ME_DRAFT_DIR" -RunDir "EDIT_ME_RUN_DIR" -Mode dry-run
```
