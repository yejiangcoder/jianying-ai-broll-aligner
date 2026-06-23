param(
  [string]$DraftDir,
  [string]$SourceImageDir,
  [string]$ReferenceBroll,
  [string]$OutDir = "",
  [string]$JyDraftc = "",
  [int]$Count = 10,
  [string]$SelectionMode = "captions",
  [switch]$Recursive
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pipeline_current_draft.ps1")
$DraftDir = Resolve-ImageAlignerDraftDir -DraftDir $DraftDir
if ([string]::IsNullOrWhiteSpace($SourceImageDir) -or [string]::IsNullOrWhiteSpace($ReferenceBroll)) {
  throw "Explicit -SourceImageDir and -ReferenceBroll are required. DraftDir may be omitted only after run_bind_current_draft.ps1 marks A-Roll QC passed."
}
foreach ($PathToCheck in @($DraftDir, $SourceImageDir, $ReferenceBroll)) {
  if (!(Test-Path -LiteralPath $PathToCheck)) {
    throw "Path does not exist: $PathToCheck"
  }
}

$Python = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (!(Test-Path -LiteralPath $Python)) {
  $Python = "python"
}

$ArgsList = @(
  (Join-Path $PSScriptRoot "src\create_test_visual_slot_package.py"),
  "--draft-dir", $DraftDir,
  "--source-image-dir", $SourceImageDir,
  "--reference-broll", $ReferenceBroll,
  "--count", ([string]$Count),
  "--selection-mode", $SelectionMode
)
if (![string]::IsNullOrWhiteSpace($OutDir)) {
  $ArgsList += @("--out-dir", $OutDir)
}
if (![string]::IsNullOrWhiteSpace($JyDraftc)) {
  if (!(Test-Path -LiteralPath $JyDraftc)) {
    throw "jy-draftc path does not exist: $JyDraftc"
  }
  $ArgsList += @("--jy-draftc", $JyDraftc)
}
if ($Recursive) {
  $ArgsList += "--recursive"
}

& $Python @ArgsList
exit $LASTEXITCODE
