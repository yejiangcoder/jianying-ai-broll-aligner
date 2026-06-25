param(
  [string]$DraftDir = "",
  [string]$TimelineId = "",
  [string]$UserScriptPath = "",
  [string]$CorrectionMapJson = "",
  [string]$RunRoot = "D:\auto_clip_runtime\subtitle_recalibration_runs",
  [string]$CurrentDraftState = "D:\auto_clip_runtime\video_pipeline\current_draft.json",
  [string]$JyDraftc = "D:\video tools\jianying-ai-image-aligner\vendor\jy-draftc-bin\jy-draftc-amd64-windows\jy-draftc.exe",
  [string]$ArollSrc = "D:\video tools\jianying-aroll-inspector\src",
  [int]$GapThresholdUs = 80000,
  [int]$MaxFillGapUs = 2000000,
  [int]$MaxLineChars = 16,
  [ValidateSet("suggest", "apply", "off")]
  [string]$ScriptPhraseRescueMode = "suggest",
  [switch]$Apply,
  [switch]$ManualRefineConfirmed,
  [switch]$AllowOpenJianying,
  [switch]$AllowMultipleTextTracks,
  [switch]$NoMaskProfanity,
  [switch]$NoScriptPhraseRescue,
  [switch]$NoLayout,
  [switch]$NoFillGaps
)

$ErrorActionPreference = "Stop"

if ($DraftDir -and -not (Test-Path -LiteralPath $DraftDir)) {
  throw "DraftDir does not exist: $DraftDir"
}
if ($UserScriptPath -and -not (Test-Path -LiteralPath $UserScriptPath)) {
  throw "UserScriptPath does not exist: $UserScriptPath"
}
if ($CorrectionMapJson -and -not (Test-Path -LiteralPath $CorrectionMapJson)) {
  throw "CorrectionMapJson does not exist: $CorrectionMapJson"
}
if (-not (Test-Path -LiteralPath $JyDraftc)) {
  throw "JyDraftc does not exist: $JyDraftc"
}
if (-not (Test-Path -LiteralPath $ArollSrc)) {
  throw "ArollSrc does not exist: $ArollSrc"
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$env:JY_DRAFTC = $JyDraftc
$env:JY_DRAFTC_EXE = $JyDraftc
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$ArollSrc;$env:PYTHONPATH" } else { $ArollSrc }

$toolPath = Join-Path (Split-Path -Parent $PSScriptRoot) "tools\manual_refined_caption_cleanup.py"
$argsList = @(
  $toolPath,
  "--run-root", $RunRoot,
  "--current-draft-state", $CurrentDraftState,
  "--jy-draftc", $JyDraftc,
  "--gap-threshold-us", $GapThresholdUs,
  "--max-fill-gap-us", $MaxFillGapUs,
  "--max-line-chars", $MaxLineChars,
  "--script-phrase-rescue-mode", $ScriptPhraseRescueMode
)

if ($DraftDir) { $argsList += @("--draft-dir", $DraftDir) }
if ($TimelineId) { $argsList += @("--timeline-id", $TimelineId) }
if ($UserScriptPath) { $argsList += @("--user-script-path", $UserScriptPath) }
if ($CorrectionMapJson) { $argsList += @("--correction-map-json", $CorrectionMapJson) }
if ($Apply) { $argsList += "--apply" }
if ($ManualRefineConfirmed) { $argsList += "--manual-refine-confirmed" }
if ($AllowOpenJianying) { $argsList += "--allow-open-jianying" }
if ($AllowMultipleTextTracks) { $argsList += "--allow-multiple-text-tracks" }
if ($NoMaskProfanity) { $argsList += "--no-mask-profanity" }
if ($NoScriptPhraseRescue) { $argsList += "--no-script-phrase-rescue" }
if ($NoLayout) { $argsList += "--no-layout" }
if ($NoFillGaps) { $argsList += "--no-fill-gaps" }

& py -3 @argsList
if ($LASTEXITCODE -ne 0) {
  throw "Manual refined caption cleanup failed with exit code $LASTEXITCODE. Inspect the printed caption_cleanup_report.json."
}
