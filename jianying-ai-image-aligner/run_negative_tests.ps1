param(
  [string]$DraftDir,
  [string]$BrollMd,
  [string]$ImageDir,
  [string]$VisualSlotPlan,
  [string]$JyDraftc = "",
  [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pipeline_current_draft.ps1")
$DraftDir = Resolve-ImageAlignerDraftDir -DraftDir $DraftDir
if ([string]::IsNullOrWhiteSpace($BrollMd) -or [string]::IsNullOrWhiteSpace($ImageDir) -or [string]::IsNullOrWhiteSpace($VisualSlotPlan)) {
  throw "Explicit -BrollMd, -ImageDir, and -VisualSlotPlan are required for negative tests."
}
foreach ($PathToCheck in @($DraftDir, $BrollMd, $ImageDir, $VisualSlotPlan)) {
  if (!(Test-Path -LiteralPath $PathToCheck)) {
    throw "Path does not exist: $PathToCheck"
  }
}
if ([string]::IsNullOrWhiteSpace($JyDraftc)) {
  if (![string]::IsNullOrWhiteSpace($env:JY_DRAFTC_EXE)) {
    $JyDraftc = $env:JY_DRAFTC_EXE
  } elseif (![string]::IsNullOrWhiteSpace($env:JY_DRAFTC)) {
    $JyDraftc = $env:JY_DRAFTC
  }
}
if ([string]::IsNullOrWhiteSpace($JyDraftc) -or !(Test-Path -LiteralPath $JyDraftc)) {
  throw "jy-draftc path does not exist. Pass -JyDraftc or set JY_DRAFTC_EXE/JY_DRAFTC."
}

$Python = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (!(Test-Path -LiteralPath $Python)) {
  $Python = "python"
}

$ArgsList = @(
  (Join-Path $PSScriptRoot "src\negative_tests.py"),
  "--draft-dir", $DraftDir,
  "--broll", $BrollMd,
  "--image-dir", $ImageDir,
  "--visual-slot-plan", $VisualSlotPlan,
  "--jy-draftc", $JyDraftc
)
if (![string]::IsNullOrWhiteSpace($OutDir)) {
  $ArgsList += @("--out-dir", $OutDir)
}

& $Python @ArgsList
exit $LASTEXITCODE
