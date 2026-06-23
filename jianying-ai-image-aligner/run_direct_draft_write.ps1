param(
  [string]$DraftDir,
  [string]$BrollMd,
  [string]$ImageDir,
  [string]$VisualSlotPlan,
  [string]$JyDraftc = "",
  [switch]$ConfirmWrite
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pipeline_current_draft.ps1")
$DraftDir = Resolve-ImageAlignerDraftDir -DraftDir $DraftDir
if ([string]::IsNullOrWhiteSpace($BrollMd) -or [string]::IsNullOrWhiteSpace($ImageDir) -or [string]::IsNullOrWhiteSpace($VisualSlotPlan)) {
  throw "Explicit -BrollMd, -ImageDir, and -VisualSlotPlan are required. DraftDir may be omitted only after run_bind_current_draft.ps1 marks A-Roll QC passed."
}
foreach ($PathToCheck in @($DraftDir, $BrollMd, $ImageDir, $VisualSlotPlan)) {
  if (!(Test-Path -LiteralPath $PathToCheck)) {
    throw "Path does not exist: $PathToCheck"
  }
}
Write-Host "CONFIRM_DRAFT_DIR=$DraftDir"
Write-Host "CONFIRM_BROLL_MD=$BrollMd"
Write-Host "CONFIRM_IMAGE_DIR=$ImageDir"
Write-Host "CONFIRM_VISUAL_SLOT_PLAN=$VisualSlotPlan"
Write-Host "CONFIRM_MODE=$(if ($ConfirmWrite) { 'WRITE_AFTER_PREFLIGHT' } else { 'PREFLIGHT_ONLY' })"
$Python = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (!(Test-Path -LiteralPath $Python)) {
  $Python = "python"
}

$WriterArgs = @(
  (Join-Path $PSScriptRoot "src\direct_draft_broll_writer.py"),
  "--draft-dir", $DraftDir,
  "--broll", $BrollMd,
  "--image-dir", $ImageDir,
  "--visual-slot-plan", $VisualSlotPlan
)
if (![string]::IsNullOrWhiteSpace($JyDraftc)) {
  if (!(Test-Path -LiteralPath $JyDraftc)) {
    throw "jy-draftc path does not exist: $JyDraftc"
  }
  $WriterArgs += @("--jy-draftc", $JyDraftc)
}

if ($ConfirmWrite) {
  & (Join-Path $PSScriptRoot "cleanup_runtime.ps1") -KeepLatest 5 -ConfirmDelete
  Get-Process JianyingPro -ErrorAction SilentlyContinue | Stop-Process -Force
  $WriterArgs += "--confirm-write"
} else {
  $WriterArgs += "--preflight-only"
}

& $Python @WriterArgs
exit $LASTEXITCODE
