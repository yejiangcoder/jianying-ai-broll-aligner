param(
  [string]$DraftDir,
  [string]$BrollMd,
  [string]$ImageDir,
  [string]$VisualSlotPlan,
  [string]$JyDraftc = ""
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
$Python = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (!(Test-Path -LiteralPath $Python)) {
  $Python = "python"
}

& (Join-Path $PSScriptRoot "cleanup_runtime.ps1") -KeepLatest 5 -ConfirmDelete

$CheckArgs = @(
  (Join-Path $PSScriptRoot "src\pipeline_contract_check.py"),
  "--draft-dir", $DraftDir,
  "--broll", $BrollMd,
  "--image-dir", $ImageDir,
  "--visual-slot-plan", $VisualSlotPlan
)
if (![string]::IsNullOrWhiteSpace($JyDraftc)) {
  if (!(Test-Path -LiteralPath $JyDraftc)) {
    throw "jy-draftc path does not exist: $JyDraftc"
  }
  $CheckArgs += @("--jy-draftc", $JyDraftc)
}

& $Python @CheckArgs
exit $LASTEXITCODE
