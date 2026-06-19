param(
  [Parameter(Mandatory = $true)]
  [string]$DraftDir,

  [string]$RunRoot = "",

  [string]$JyDraftc = "",

  [string]$BaselinePath = "",

  [ValidateSet("auto", "initial-clean-cluster", "latest-clean-before-dirty")]
  [string]$SelectionMode = "auto",

  [int]$InitialClusterSeconds = 600,

  [switch]$Apply,

  [switch]$StopJianying,

  [switch]$Force,

  [switch]$NoQuarantineDirtyBackups,

  [switch]$IgnoreBaselineRegistry,

  [switch]$KeepDecrypted
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Get-DefaultRuntimeRoot {
  if ($env:AUTO_CLIP_RUNTIME_DIR) {
    return [string]$env:AUTO_CLIP_RUNTIME_DIR
  }
  return (Join-Path $HOME ".auto_clip_runtime")
}

function Resolve-ExistingPath([string]$PathValue, [string]$Label) {
  if (-not $PathValue) {
    throw "$Label is empty"
  }
  if (-not (Test-Path -LiteralPath $PathValue)) {
    throw "$Label does not exist: $PathValue"
  }
  return (Resolve-Path -LiteralPath $PathValue).Path
}

function Get-DefaultJyDraftc {
  if ($env:JY_DRAFTC -and (Test-Path -LiteralPath $env:JY_DRAFTC)) {
    return [string]$env:JY_DRAFTC
  }
  if ($env:JY_DRAFTC_EXE -and (Test-Path -LiteralPath $env:JY_DRAFTC_EXE)) {
    return [string]$env:JY_DRAFTC_EXE
  }
  return ""
}

function Invoke-RepoPython([string[]]$Arguments) {
  $py = Get-Command py.exe -ErrorAction SilentlyContinue
  if ($py) {
    & $py.Source -3 @Arguments
    if ($LASTEXITCODE -ne 0) {
      throw "Python command failed with exit code $LASTEXITCODE"
    }
    return
  }

  $python = Get-Command python.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.Source -and $_.Source -notlike "*\WindowsApps\python.exe" } |
    Select-Object -First 1
  if ($python) {
    & $python.Source @Arguments
    if ($LASTEXITCODE -ne 0) {
      throw "Python command failed with exit code $LASTEXITCODE"
    }
    return
  }

  throw "No usable Python found. Install py launcher or set PATH to a real python.exe."
}

function Read-JsonFile([string]$PathValue) {
  if (-not (Test-Path -LiteralPath $PathValue)) {
    throw "JSON file missing: $PathValue"
  }
  return (Get-Content -LiteralPath $PathValue -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Assert-PathInside([string]$Child, [string]$Parent, [string]$Message) {
  $childResolved = [System.IO.Path]::GetFullPath($Child)
  $parentResolved = [System.IO.Path]::GetFullPath($Parent)
  if (-not $childResolved.StartsWith($parentResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "$Message Child=$childResolved Parent=$parentResolved"
  }
}

function Get-RelativePathCompat([string]$Parent, [string]$Child) {
  $parentResolved = [System.IO.Path]::GetFullPath($Parent).TrimEnd([char[]]@('\', '/'))
  $childResolved = [System.IO.Path]::GetFullPath($Child)
  if (-not $childResolved.StartsWith($parentResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Child path is not inside parent. Child=$childResolved Parent=$parentResolved"
  }
  return $childResolved.Substring($parentResolved.Length).TrimStart([char[]]@('\', '/'))
}

function Get-DraftKeyHash([string]$DraftPath) {
  $draftKeySource = [System.Text.Encoding]::UTF8.GetBytes($DraftPath.ToLowerInvariant())
  $sha = [System.Security.Cryptography.SHA256]::Create()
  return ([System.BitConverter]::ToString($sha.ComputeHash($draftKeySource))).Replace("-", "").Substring(0, 12)
}

function Get-BaselineRegistrySlot([string]$RegistryRoot, [string]$DraftPath) {
  $draftLeaf = Split-Path -Leaf $DraftPath
  $safeDraftLeaf = ($draftLeaf -replace '[^A-Za-z0-9_.\-\u4e00-\u9fa5]', '_')
  return Join-Path $RegistryRoot ($safeDraftLeaf + "_" + (Get-DraftKeyHash $DraftPath))
}

function Copy-ToQuarantine([string]$Source, [string]$SourceRoot, [string]$QuarantineRoot, [string]$Bucket) {
  if (-not (Test-Path -LiteralPath $Source)) {
    return $null
  }
  Assert-PathInside -Child $Source -Parent $SourceRoot -Message "Refusing to preserve a file outside the expected source root."
  $relative = Get-RelativePathCompat -Parent $SourceRoot -Child $Source
  $destination = Join-Path $QuarantineRoot (Join-Path $Bucket $relative)
  Assert-PathInside -Child $destination -Parent $QuarantineRoot -Message "Refusing unexpected quarantine destination."
  New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
  Copy-Item -LiteralPath $Source -Destination $destination -Force
  return [pscustomobject]@{ From = $Source; To = $destination; Action = "copy" }
}

function Move-ToQuarantine([string]$Source, [string]$SourceRoot, [string]$QuarantineRoot, [string]$Bucket) {
  if (-not (Test-Path -LiteralPath $Source)) {
    return $null
  }
  Assert-PathInside -Child $Source -Parent $SourceRoot -Message "Refusing to move a file outside the expected source root."
  $relative = Get-RelativePathCompat -Parent $SourceRoot -Child $Source
  $destination = Join-Path $QuarantineRoot (Join-Path $Bucket $relative)
  Assert-PathInside -Child $destination -Parent $QuarantineRoot -Message "Refusing unexpected quarantine destination."
  New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
  Move-Item -LiteralPath $Source -Destination $destination -Force
  return [pscustomobject]@{ From = $Source; To = $destination; Action = "move" }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$analyzer = Join-Path $repoRoot "tools\jianying_draft_rollback_analyzer.py"
if (-not (Test-Path -LiteralPath $analyzer)) {
  throw "Rollback analyzer missing: $analyzer"
}

$draftDirResolved = Resolve-ExistingPath $DraftDir "DraftDir"
if (-not $JyDraftc) {
  $JyDraftc = Get-DefaultJyDraftc
}
$jyDraftcResolved = Resolve-ExistingPath $JyDraftc "JyDraftc"
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
  $RunRoot = Join-Path (Get-DefaultRuntimeRoot) "draft_rollback_runs"
}
$backupRoot = Join-Path $draftDirResolved ".backup"
if (-not (Test-Path -LiteralPath $backupRoot)) {
  throw "Draft backup directory does not exist: $backupRoot"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $RunRoot ("draft_rollback_" + $timestamp)
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$runDirResolved = (Resolve-Path -LiteralPath $runDir).Path

$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:JY_DRAFTC = $jyDraftcResolved
$env:JY_DRAFTC_EXE = $jyDraftcResolved

$baselineRegistry = Join-Path $RunRoot "_rollback_baselines"
if (-not $BaselinePath -and -not $IgnoreBaselineRegistry) {
  $baselineSlotProbe = Get-BaselineRegistrySlot -RegistryRoot $baselineRegistry -DraftPath $draftDirResolved
  $baselineManifestProbe = Join-Path $baselineSlotProbe "baseline_manifest.json"
  if (Test-Path -LiteralPath $baselineManifestProbe) {
    $baselineManifest = Read-JsonFile $baselineManifestProbe
    $registeredBaseline = [string]$baselineManifest.baseline_path
    if ($registeredBaseline -and (Test-Path -LiteralPath $registeredBaseline)) {
      $BaselinePath = $registeredBaseline
    }
  }
}

$pythonArgs = @(
  $analyzer,
  "--draft-dir", $draftDirResolved,
  "--jy-draftc", $jyDraftcResolved,
  "--output-dir", $runDirResolved,
  "--selection-mode", $SelectionMode,
  "--initial-cluster-seconds", ([string]$InitialClusterSeconds)
)
if ($BaselinePath) {
  $baselineResolved = Resolve-ExistingPath $BaselinePath "BaselinePath"
  $pythonArgs += @("--baseline-path", $baselineResolved)
}
if ($KeepDecrypted) {
  $pythonArgs += "--keep-decrypted"
}

Push-Location $repoRoot
try {
  Invoke-RepoPython $pythonArgs
} finally {
  Pop-Location
}

$reportPath = Join-Path $runDirResolved "rollback_candidate_report.json"
$report = Read-JsonFile $reportPath
$selected = $report.selected_candidate
if ($null -eq $selected -or -not $selected.path) {
  throw "No clean rollback baseline selected. See report: $reportPath"
}

$summary = [ordered]@{
  APPLY = [bool]$Apply
  RUN_DIR = $runDirResolved
  REPORT_PATH = $reportPath
  DRAFT_DIR = $draftDirResolved
  SELECTED_BASELINE = [string]$selected.path
  SELECTED_SHA256 = [string]$selected.sha256
  SELECTION_CONFIDENCE = [string]$report.selection.confidence
  SELECTION_WARNINGS = @($report.selection.warnings)
  CLEAN_CANDIDATE_COUNT = [int]$report.clean_candidate_count
  DIRTY_CANDIDATE_COUNT = [int]$report.dirty_candidate_count
  ACTIVE_DIRTY_TARGET_COUNT = [int]$report.active_dirty_target_count
  QUARANTINE_BACKUP_COUNT = [int]$report.quarantine_backup_count
  SELECTION_MODE = $SelectionMode
}

if (-not $Apply) {
  $summary.ROLLBACK_DONE = $false
  $summary.NEXT_COMMAND = "scripts/rollback_jianying_draft.ps1 -DraftDir `"$draftDirResolved`" -Apply -StopJianying"
  $summary | ConvertTo-Json -Depth 20
  return
}

if ($report.selection.confidence -eq "low" -and -not $Force) {
  throw "Rollback baseline confidence is low. Inspect $reportPath or rerun with -BaselinePath / -Force."
}

$processes = @(Get-Process | Where-Object { $_.ProcessName -match "Jianying|CapCut|jianying|capcut" })
if ($processes.Count -gt 0) {
  if (-not $StopJianying) {
    $names = ($processes | ForEach-Object { "$($_.ProcessName):$($_.Id)" }) -join ", "
    throw "Jianying/CapCut processes are running. Rerun with -StopJianying to stop them before rollback. Processes: $names"
  }
  foreach ($process in $processes) {
    Stop-Process -Id $process.Id -Force
  }
  Start-Sleep -Milliseconds 500
}

$quarantineRoot = Join-Path $runDirResolved "quarantine_before_rollback"
New-Item -ItemType Directory -Force -Path $quarantineRoot | Out-Null
$quarantineRootResolved = (Resolve-Path -LiteralPath $quarantineRoot).Path
Assert-PathInside -Child $quarantineRootResolved -Parent $runDirResolved -Message "Unexpected quarantine root."

$baselineCopy = Join-Path $runDirResolved "selected_clean_baseline.enc.json"
Copy-Item -LiteralPath ([string]$selected.path) -Destination $baselineCopy -Force
$baselineHash = (Get-FileHash -LiteralPath $baselineCopy -Algorithm SHA256).Hash
if ($baselineHash -ne [string]$selected.sha256) {
  throw "Selected baseline hash changed while preparing rollback."
}

$preservedActive = @()
foreach ($target in @($report.active_targets)) {
  $targetPath = [string]$target.path
  if ($targetPath) {
    $item = Copy-ToQuarantine -Source $targetPath -SourceRoot $draftDirResolved -QuarantineRoot $quarantineRootResolved -Bucket "active_before_rollback"
    if ($null -ne $item) {
      $preservedActive += $item
    }
  }
}

$movedBackups = @()
if (-not $NoQuarantineDirtyBackups) {
  foreach ($backupPath in @($report.quarantine_backup_paths)) {
    if ($backupPath) {
      $item = Move-ToQuarantine -Source ([string]$backupPath) -SourceRoot $backupRoot -QuarantineRoot $quarantineRootResolved -Bucket "dirty_backup_entries"
      if ($null -ne $item) {
        $movedBackups += $item
      }
    }
  }
}

$writtenTargets = @()
foreach ($target in @($report.active_targets)) {
  $targetPath = [string]$target.path
  if (-not $targetPath) {
    continue
  }
  Assert-PathInside -Child $targetPath -Parent $draftDirResolved -Message "Refusing to write outside DraftDir."
  New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
  Copy-Item -LiteralPath $baselineCopy -Destination $targetPath -Force
  $targetHash = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash
  $writtenTargets += [pscustomobject]@{
    Path = $targetPath
    SHA256 = $targetHash
    MatchesBaseline = ($targetHash -eq $baselineHash)
  }
}

$verifyDir = Join-Path $runDirResolved "verify_after_rollback"
New-Item -ItemType Directory -Force -Path $verifyDir | Out-Null
$verifyArgs = @(
  $analyzer,
  "--draft-dir", $draftDirResolved,
  "--jy-draftc", $jyDraftcResolved,
  "--output-dir", $verifyDir,
  "--baseline-path", $baselineCopy
)
if ($KeepDecrypted) {
  $verifyArgs += "--keep-decrypted"
}
Push-Location $repoRoot
try {
  Invoke-RepoPython $verifyArgs
} finally {
  Pop-Location
}

$verifyReportPath = Join-Path $verifyDir "rollback_candidate_report.json"
$verifyReport = Read-JsonFile $verifyReportPath
$hashMismatch = @($writtenTargets | Where-Object { -not $_.MatchesBaseline })
$postDirtyTargets = @($verifyReport.active_targets | Where-Object { [int]$_.automation_marker_count -gt 0 })

if ($hashMismatch.Count -gt 0) {
  throw "Rollback verification failed: one or more active targets do not match selected baseline hash."
}
if ($postDirtyTargets.Count -gt 0) {
  throw "Rollback verification failed: one or more active targets still contain automation markers."
}

New-Item -ItemType Directory -Force -Path $baselineRegistry | Out-Null
$baselineSlot = Get-BaselineRegistrySlot -RegistryRoot $baselineRegistry -DraftPath $draftDirResolved
New-Item -ItemType Directory -Force -Path $baselineSlot | Out-Null
$registryBaselineCopy = Join-Path $baselineSlot ("baseline_" + $baselineHash.Substring(0, 12) + ".enc.json")
Copy-Item -LiteralPath $baselineCopy -Destination $registryBaselineCopy -Force
$registryManifest = [ordered]@{
  draft_dir = $draftDirResolved
  baseline_sha256 = $baselineHash
  baseline_path = $registryBaselineCopy
  selected_source_path = [string]$selected.path
  created_at = (Get-Date).ToString("s")
  run_dir = $runDirResolved
  selection_mode = $SelectionMode
}
$registryManifestPath = Join-Path $baselineSlot "baseline_manifest.json"
$registryManifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $registryManifestPath -Encoding UTF8

$result = [ordered]@{
  ROLLBACK_DONE = $true
  RUN_DIR = $runDirResolved
  REPORT_PATH = $reportPath
  VERIFY_REPORT_PATH = $verifyReportPath
  DRAFT_DIR = $draftDirResolved
  SELECTED_BASELINE = [string]$selected.path
  SELECTED_SHA256 = $baselineHash
  SELECTION_CONFIDENCE = [string]$report.selection.confidence
  SELECTION_WARNINGS = @($report.selection.warnings)
  PRESERVED_ACTIVE_FILE_COUNT = $preservedActive.Count
  MOVED_DIRTY_BACKUP_COUNT = $movedBackups.Count
  WRITTEN_TARGET_COUNT = $writtenTargets.Count
  ALL_TARGETS_MATCH_BASELINE = $true
  ACTIVE_DIRTY_TARGET_COUNT_AFTER = 0
  ACTIVE_VIDEO_SEGMENTS_AFTER = @($verifyReport.active_targets | Where-Object { $_.path -like "*draft_content.json" } | Select-Object -First 1 -ExpandProperty video_segments)
  ACTIVE_TEXT_SEGMENTS_AFTER = @($verifyReport.active_targets | Where-Object { $_.path -like "*draft_content.json" } | Select-Object -First 1 -ExpandProperty text_segments)
  QUARANTINE_DIR = $quarantineRootResolved
  BASELINE_REGISTRY_MANIFEST = $registryManifestPath
}

$resultPath = Join-Path $runDirResolved "rollback_result.json"
$result | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath $resultPath -Encoding UTF8
$result.RESULT_PATH = $resultPath
$result | ConvertTo-Json -Depth 50
