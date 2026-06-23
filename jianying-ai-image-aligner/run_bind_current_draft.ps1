param(
  [string]$DraftDir,
  [string]$JyDraftc = "",
  [string]$StatePath = "",
  [string]$Stage = "aroll_written"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($DraftDir)) {
  throw "Explicit -DraftDir is required when binding the QC-passed draft."
}
if (!(Test-Path -LiteralPath $DraftDir)) {
  throw "DraftDir does not exist: $DraftDir"
}

$Python = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (!(Test-Path -LiteralPath $Python)) {
  $Python = "python"
}

$ArgsList = @(
  (Join-Path $PSScriptRoot "src\bind_current_draft.py"),
  "--draft-dir", $DraftDir,
  "--stage", $Stage
)
if (![string]::IsNullOrWhiteSpace($JyDraftc)) {
  if (!(Test-Path -LiteralPath $JyDraftc)) {
    throw "jy-draftc path does not exist: $JyDraftc"
  }
  $ArgsList += @("--jy-draftc", $JyDraftc)
}
if (![string]::IsNullOrWhiteSpace($StatePath)) {
  $ArgsList += @("--state-path", $StatePath)
}

& $Python @ArgsList
exit $LASTEXITCODE
