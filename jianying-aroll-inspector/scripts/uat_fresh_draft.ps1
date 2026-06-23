param(
  [string]$DraftDir = "",
  [string]$RunRoot = "D:\auto_clip_runtime\aroll_v21_uat_runs",
  [ValidateSet("auto", "deterministic-baseline", "semantic-requests-only", "deepseek", "fail-closed", "default")]
  [string]$SemanticMode = "auto",
  [string]$ReadyRunDir = "",
  [ValidateSet("minimal", "standard", "debug")]
  [string]$ReportProfile = "standard",
  [switch]$Commit
)

$ErrorActionPreference = "Stop"

function Read-JsonFile([string]$Path) {
  $resolvedPath = $Path
  if (-not (Test-Path -LiteralPath $resolvedPath)) {
    $gzPath = "$Path.gz"
    if (Test-Path -LiteralPath $gzPath) {
      $resolvedPath = $gzPath
    }
  }
  if (-not (Test-Path -LiteralPath $resolvedPath)) {
    throw "Required report missing: $Path"
  }
  $helper = @'
import gzip
import json
import sys


def normalize(obj):
    if isinstance(obj, dict):
        grouped = {}
        for key, value in obj.items():
            normalized_key = str(key)
            lower_key = normalized_key.lower()
            grouped.setdefault(lower_key, []).append((normalized_key, normalize(value)))

        out = {}
        for lower_key, items in grouped.items():
            upper_items = [item for item in items if item[0].upper() == item[0]]
            if upper_items:
                normalized_key, normalized_value = upper_items[-1]
            else:
                normalized_key, normalized_value = items[-1]
            out[normalized_key] = normalized_value
        return out
    if isinstance(obj, list):
        return [normalize(value) for value in obj]
    return obj


path = sys.argv[1]
open_fn = gzip.open if path.endswith(".gz") else open
with open_fn(path, "rt", encoding="utf-8") as f:
    payload = json.load(f)

print(json.dumps(normalize(payload), ensure_ascii=False))
'@
  $tmp = Join-Path $env:TEMP "read_json_case_safe.py"
  Set-Content -Path $tmp -Value $helper -Encoding UTF8
  try {
    $jsonText = & py -3 $tmp $resolvedPath
    return $jsonText | ConvertFrom-Json
  } finally {
    Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
  }
}

function Read-JsonValue([object]$Object, [string[]]$Keys) {
  foreach ($key in $Keys) {
    if ($null -eq $Object) {
      continue
    }
    if ($Object -is [System.Collections.IDictionary] -and $Object.Contains($key)) {
      return $Object[$key]
    }
    if ($Object.PSObject.Properties.Match($key).Count -gt 0) {
      return $Object.$key
    }
  }
  return $null
}

function Assert-True([bool]$Condition, [string]$Message) {
  if (-not $Condition) {
    throw $Message
  }
}

function Resolve-PinnedJianyingInstallDir {
  $pinRoot = [string]$env:JY_VERSION_PIN_ROOT
  if (-not $pinRoot) {
    $pinRoot = "D:\auto_clip_runtime\jianying_version_pin"
  }
  if (-not (Test-Path -LiteralPath $pinRoot)) {
    return ""
  }

  $manifests = @(Get-ChildItem -LiteralPath $pinRoot -Filter "pin_manifest.json" -Recurse -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending)
  foreach ($manifest in $manifests) {
    try {
      $pin = Get-Content -LiteralPath $manifest.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
      continue
    }
    $installDir = [string]$pin.pinned_install_dir
    if ($installDir -and (Test-Path -LiteralPath (Join-Path $installDir "videoeditor.dll"))) {
      return $installDir
    }
  }
  return ""
}

function Resolve-JianyingInstallDir {
  $configured = [string]$env:JY_INSTALL_DIR
  if ($configured -and (Test-Path -LiteralPath (Join-Path $configured "videoeditor.dll"))) {
    return $configured
  }

  $pinned = Resolve-PinnedJianyingInstallDir
  if ($pinned) {
    return $pinned
  }

  $roots = @()
  if ($env:JY_INSTALL_ROOT) {
    $roots += [string]$env:JY_INSTALL_ROOT
  }
  if ($configured) {
    $parent = Split-Path -Parent $configured
    if ($parent) {
      $roots += $parent
    }
  }
  $roots += "D:\JianyingPro"
  $roots = $roots | Where-Object { $_ } | Select-Object -Unique

  $candidates = @()
  foreach ($root in $roots) {
    if (-not (Test-Path -LiteralPath $root)) {
      continue
    }
    $candidates += Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
      Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "videoeditor.dll") } |
      ForEach-Object {
        $version = [version]"0.0.0.0"
        [void][version]::TryParse($_.Name, [ref]$version)
        [pscustomobject]@{
          Path = $_.FullName
          Version = $version
          DllLastWriteTime = (Get-Item -LiteralPath (Join-Path $_.FullName "videoeditor.dll")).LastWriteTime
        }
      }
  }

  $selected = $candidates | Sort-Object Version, DllLastWriteTime -Descending | Select-Object -First 1
  if ($null -eq $selected) {
    throw "Unable to resolve JianyingPro install dir with videoeditor.dll. Checked roots: $($roots -join ', ')"
  }
  return [string]$selected.Path
}

function Sync-JyDraftcEnv {
  $jyDraftc = [string]$env:JY_DRAFTC
  if (-not $jyDraftc) {
    $jyDraftc = [string]$env:JY_DRAFTC_EXE
  }
  if (-not $jyDraftc) {
    return
  }
  if (-not (Test-Path -LiteralPath $jyDraftc)) {
    throw "JY_DRAFTC does not exist: $jyDraftc"
  }

  $installDir = Resolve-JianyingInstallDir
  $env:JY_INSTALL_DIR = $installDir
  $envFile = Join-Path (Split-Path -Parent $jyDraftc) ".env"
  $desired = "JY_INSTALL_DIR=$installDir"
  $current = ""
  if (Test-Path -LiteralPath $envFile) {
    $current = (Get-Content -LiteralPath $envFile -Raw).Trim()
  }
  if ($current -ne $desired) {
    Set-Content -LiteralPath $envFile -Value $desired -Encoding ASCII
  }
}

if ([string]::IsNullOrWhiteSpace($DraftDir)) {
  throw "DraftDir is required. Pass an explicit disposable V21 draft path."
}
if (-not (Test-Path -LiteralPath $DraftDir)) {
  throw "DraftDir does not exist: $DraftDir"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repoRoot "src"
if (-not $env:REFERENCE_VIDEO_DATA_CATCHER_DEEPSEEK_CONFIG_PATH) {
  $referenceDeepSeekConfig = "D:\idea-project\videoDataCatcher\src\main\resources\application.yaml"
  if (Test-Path -LiteralPath $referenceDeepSeekConfig) {
    $env:REFERENCE_VIDEO_DATA_CATCHER_DEEPSEEK_CONFIG_PATH = $referenceDeepSeekConfig
  }
}
Sync-JyDraftcEnv
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = ""
if (-not [string]::IsNullOrWhiteSpace($ReadyRunDir)) {
  $runDir = $ReadyRunDir
  if (-not (Test-Path -LiteralPath $runDir)) {
    throw "ReadyRunDir does not exist: $runDir"
  }
} else {
  $runDir = Join-Path $RunRoot ("v21_fresh_draft_uat_" + $timestamp)
  New-Item -ItemType Directory -Force -Path $runDir | Out-Null

  $dryArgs = @(
    "-m", "aroll_v21.cli",
    "--mode", "dry-run",
    "--draft-dir", $DraftDir,
    "--output-dir", $runDir,
    "--semantic-mode", $SemanticMode,
    "--report-profile", $ReportProfile
  )

  Push-Location $repoRoot
  try {
    & py -3 @dryArgs
    if ($LASTEXITCODE -ne 0) {
      throw "V21 dry-run command failed with exit code $LASTEXITCODE"
    }
  } finally {
    Pop-Location
  }
}

$summary = Read-JsonFile (Join-Path $runDir "run_summary.json")
$prewrite = Read-JsonFile (Join-Path $runDir "prewrite_report.json")
$quality = Read-JsonFile (Join-Path $runDir "quality_gate_report.json")
$finalTimeline = Read-JsonFile (Join-Path $runDir "final_timeline.json")
$captions = Read-JsonFile (Join-Path $runDir "captions.json")
$resolvedMap = Read-JsonValue -Object $prewrite -Keys @("resolved_template_map")
$resolvedCount = 0
if ($null -ne $resolvedMap) {
  if ($resolvedMap -is [System.Collections.ICollection]) {
    $resolvedCount = $resolvedMap.Count
  } elseif ($resolvedMap -is [System.Collections.IDictionary]) {
    $resolvedCount = $resolvedMap.Count
  } elseif ($resolvedMap.PSObject.Properties) {
    $resolvedCount = ($resolvedMap.PSObject.Properties | Measure-Object).Count
  }
}
$rough = Read-JsonValue -Object $summary -Keys @("rough_cut_quality")

Assert-True ($summary.status -eq "ok") "dry-run status is not ok"
Assert-True ([bool]$summary.READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT) "READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT is false"
Assert-True ($resolvedCount -eq $finalTimeline.Count) "resolved_template_map_count does not match final_timeline count"
Assert-True ([int]$rough.segments_lt_300ms -eq 0) "segments_lt_300ms is not zero"
Assert-True ([int]$rough.one_char_captions -eq 0) "one_char_captions is not zero"
Assert-True ([bool](Read-JsonValue -Object $prewrite -Keys @("speed_safe")) ) "speed_safe is false"
Assert-True ([bool](Read-JsonValue -Object $quality -Keys @("gate_passed")) ) "quality gate failed"
Assert-True ([bool](Read-JsonValue -Object $summary -Keys @("effective_speed_gate_passed")) ) "effective_speed_gate_passed is false"
Assert-True ([int](Read-JsonValue -Object $summary -Keys @("effective_speed_drift_count")) -eq 0) "effective_speed_drift_count is not zero"
Assert-True ([bool](Read-JsonValue -Object $summary -Keys @("final_repeat_convergence_gate_passed")) ) "final_repeat_convergence_gate_passed is false"
Assert-True ([int](Read-JsonValue -Object $summary -Keys @("final_repeat_high_count_after_convergence")) -eq 0) "final_repeat_high_count_after_convergence is not zero"
Assert-True ([bool](Read-JsonValue -Object $summary -Keys @("visual_pacing_gate_passed")) ) "visual_pacing_gate_passed is false"
Assert-True ([bool](Read-JsonValue -Object $summary -Keys @("caption_alignment_gate_passed")) ) "caption_alignment_gate_passed is false"
Assert-True ([bool](Read-JsonValue -Object $prewrite -Keys @("effect_policy_safe")) ) "effect_policy_safe is false"
Assert-True (-not [bool](Read-JsonValue -Object $prewrite -Keys @("root_mirror_check_failed"))) "root mirror check failed"

$writeSummary = $null
  $writeback = Read-JsonFile (Join-Path $runDir "writeback_report.json")
  if ($Commit) {
  $writeRunDir = Join-Path $RunRoot ("v21_fresh_draft_write_" + $timestamp)
  New-Item -ItemType Directory -Force -Path $writeRunDir | Out-Null
  $writeArgs = @(
    "-m", "aroll_v21.cli",
    "--mode", "write",
    "--draft-dir", $DraftDir,
    "--output-dir", $writeRunDir,
    "--semantic-mode", $SemanticMode,
    "--report-profile", $ReportProfile,
    "--ready-run-dir", $runDir,
    "--allow-sacrificial-write-without-postwrite-decrypt",
    "--commit"
  )
  Push-Location $repoRoot
  try {
    & py -3 @writeArgs
    if ($LASTEXITCODE -ne 0) {
      throw "V21 sacrificial write command failed with exit code $LASTEXITCODE"
    }
  } finally {
    Pop-Location
  }
  $writeSummary = Read-JsonFile (Join-Path $writeRunDir "run_summary.json")
  $writeback = Read-JsonFile (Join-Path $writeRunDir "writeback_report.json")
  $writeSuccess = Read-JsonValue -Object $writeSummary -Keys @("WRITE_SUCCESS", "write_success")
  $encryptSuccess = Read-JsonValue -Object $writeSummary -Keys @("ENCRYPT_SUCCESS", "encrypt_success")
  $commitPerformed = Read-JsonValue -Object $writeSummary -Keys @("commit_performed")
  $writebackSuccess = Read-JsonValue -Object $writeSummary -Keys @("writeback_success")
  $readyForQc = Read-JsonValue -Object $writeSummary -Keys @("ready_for_user_manual_qc")
  $onlySpecified = Read-JsonValue -Object $writeSummary -Keys @("only_specified_draft_written")
  Assert-True ([bool]$writeSuccess) "WRITE_SUCCESS is false"
  Assert-True ([bool]$encryptSuccess) "ENCRYPT_SUCCESS is false"
  Assert-True ([bool]$commitPerformed) "commit_performed is false"
  Assert-True ([bool]$writebackSuccess) "writeback_success is false"
  Assert-True ([bool]$readyForQc) "ready_for_user_manual_qc is false"
  Assert-True ([bool]$onlySpecified) "only_specified_draft_written is false"
}

$auditSummary = if ($writeSummary) { $writeSummary } else { $summary }

$summaryLines = [ordered]@{
  DRAFT_DIR = $DraftDir
  RUN_DIR = $runDir
  DRY_RUN_STATUS = $summary.status
  READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT = $summary.READY_FOR_DISPOSABLE_WRITE_PRE_AUDIT
  SPEECH_TIMELINE_PROVIDER = $summary.speech_timeline_provider
  SPEECH_TIMELINE_GRANULARITY = $summary.speech_timeline_granularity
  WORD_COUNT = $summary.word_timeline_count
  FINAL_TIMELINE_SEGMENT_COUNT = $finalTimeline.Count
  CAPTION_COUNT = $captions.Count
  RESOLVED_TEMPLATE_MAP_COUNT = $resolvedCount
  PRIMARY_VIDEO_TRACK_ID = $prewrite.primary_video_track_id
  PRIMARY_VIDEO_SEGMENT_COUNT = $prewrite.primary_video_candidate_count
  SPEED_SAFE = $prewrite.speed_safe
  DETECTED_SPEEDS = (($prewrite.detected_speeds | ForEach-Object { [string]$_ }) -join ",")
  EFFECTIVE_SPEED_GATE_PASSED = $summary.effective_speed_gate_passed
  EFFECTIVE_SPEED_MIN = $summary.effective_speed_min
  EFFECTIVE_SPEED_MAX = $summary.effective_speed_max
  EFFECTIVE_SPEED_DRIFT_COUNT = $summary.effective_speed_drift_count
  FINAL_REPEAT_CONVERGENCE_GATE_PASSED = $summary.final_repeat_convergence_gate_passed
  FINAL_REPEAT_HIGH_COUNT_AFTER_CONVERGENCE = $summary.final_repeat_high_count_after_convergence
  FINAL_REPEAT_DROPPED_SEGMENT_COUNT = $summary.final_repeat_dropped_segment_count
  DROPPED_SEGMENT_INDICES = (($summary.dropped_segment_indices | ForEach-Object { [string]$_ }) -join ",")
  DROPPED_CLUSTER_IDS = (($summary.dropped_cluster_ids | ForEach-Object { [string]$_ }) -join ",")
  FINAL_CAPTION_VISIBLE_REPEAT_GATE_PASSED = $summary.final_caption_visible_repeat_gate_passed
  VISIBLE_REPEAT_CANDIDATE_COUNT = $summary.visible_repeat_candidate_count
  CONTAINMENT_REPEAT_COUNT = $summary.containment_repeat_count
  PREFIX_SUFFIX_OVERLAP_COUNT = $summary.prefix_suffix_overlap_count
  NGRAM_REPEAT_COUNT = $summary.ngram_repeat_count
  NEAR_DUPLICATE_VISIBLE_CAPTION_COUNT = $summary.near_duplicate_visible_caption_count
  VISUAL_PACING_GATE_PASSED = $summary.visual_pacing_gate_passed
  VISUAL_PACING_EXECUTED = $summary.visual_pacing_executed
  VISUAL_PACING_MERGED_COUNT = $summary.visual_pacing_merged_count
  VISUAL_MERGE_SAFETY_GATE_PASSED = $summary.visual_merge_safety_gate_passed
  UNSAFE_MERGE_GROUP_COUNT = $summary.unsafe_merge_group_count
  DROPPED_CONTENT_REINTRODUCED_COUNT = $summary.dropped_content_reintroduced_count
  MAX_BRIDGED_GAP_US = $summary.max_bridged_gap_us
  TOTAL_BRIDGED_GAP_US = $summary.total_bridged_gap_us
  UNSPOKEN_BRIDGE_RATIO = $summary.unspoken_bridge_ratio
  VISUAL_SHORT_SEGMENT_COUNT_LT_1200MS = $summary.visual_short_segment_count_lt_1200ms
  VISUAL_SHORT_SEGMENT_COUNT_LT_1200MS_BEFORE = $summary.visual_short_segment_count_lt_1200ms_before
  VISUAL_SHORT_SEGMENT_COUNT_LT_1200MS_AFTER = $summary.visual_short_segment_count_lt_1200ms_after
  VISUAL_SHORT_SEGMENT_COUNT_LT_1200MS_AFTER_BLOCKING = $summary.visual_short_segment_count_lt_1200ms_after_blocking
  SEMANTIC_BRIDGE_SHORT_SEGMENT_COUNT = $summary.semantic_bridge_short_segment_count
  SEMANTIC_BRIDGE_CAP = $summary.semantic_bridge_cap
  SEMANTIC_BRIDGE_REASON_COUNTS = if ($summary.semantic_bridge_reason_counts) { (($summary.semantic_bridge_reason_counts.PSObject.Properties | ForEach-Object { "$($_.Name):$($_.Value)" }) -join ",") } else { "" }
  SEMANTIC_BRIDGE_SAFE_MERGE_CANDIDATE_COUNT = $summary.semantic_bridge_safe_merge_candidate_count
  CUTS_PER_MINUTE = $summary.cuts_per_minute
  MAX_CUTS_IN_5S = $summary.max_cuts_in_5s
  BURST_CUT_COUNT = $summary.burst_cut_count
  CUT_DENSITY_GATE_ENABLED = $summary.cut_density_gate_enabled
  CUT_DENSITY_GATE_PASSED = $summary.cut_density_gate_passed
  HIDDEN_REPEAT_CLEANUP_DROPPED_WORD_COUNT = $summary.hidden_repeat_cleanup_dropped_word_count
  BOUNDARY_OVERLAP_CLEANUP_DROPPED_WORD_COUNT = $summary.boundary_overlap_cleanup_dropped_word_count
  MEDIAN_SEGMENT_DURATION_US = $summary.median_segment_duration_us
  P10_SEGMENT_DURATION_US = $summary.p10_segment_duration_us
  CAPTION_PER_VIDEO_SEGMENT_RATIO = $summary.caption_per_video_segment_ratio
  CAPTION_ALIGNMENT_GATE_PASSED = $summary.caption_alignment_gate_passed
  CAPTION_GUI_TRACK_GATE_PASSED = $summary.caption_gui_track_gate_passed
  SUBTITLE_READABILITY_GATE_PASSED = $summary.subtitle_readability_gate_passed
  VISIBLE_CAPTION_TRACK_COUNT = $summary.visible_caption_track_count
  CAPTION_LANE_COUNT = $summary.caption_lane_count
  ORPHAN_CAPTION_COUNT = $summary.orphan_caption_count
  FLOATING_CAPTION_COUNT = $summary.floating_caption_count
  CAPTION_OUTSIDE_VIDEO_COUNT = $summary.caption_outside_video_count
  CAPTION_OVERLAP_COUNT = $summary.caption_overlap_count
  CAPTION_TOO_SHORT_COUNT = $summary.caption_too_short_count
  ONE_CHAR_CAPTION_COUNT = $summary.one_char_caption_count
  CAPTION_WITHOUT_VIDEO_CONTAINER_COUNT = $summary.caption_without_video_container_count
  CAPTIONS_LE_3_CHARS = $summary.captions_le_3_chars
  CAPTIONS_LE_3_CHARS_CAP = $summary.captions_le_3_chars_cap
  SUBTITLE_INTERVAL_TOO_SHORT_COUNT = $summary.subtitle_interval_too_short_count
  SUBTITLE_INTERVAL_TOO_LONG_COUNT = $summary.subtitle_interval_too_long_count
  SUBTITLE_HARD_MAX_CHAR_COUNT = $summary.subtitle_hard_max_char_count
  CAPTION_DENSITY_PER_MINUTE = $summary.caption_density_per_minute
  MAX_CAPTIONS_IN_5S = $summary.max_captions_in_5s
  CAPTION_BURST_DENSITY_COUNT = $summary.caption_burst_density_count
  EFFECT_POLICY_SAFE = $prewrite.effect_policy_safe
  EMBEDDED_EFFECTS_PRESERVED = $prewrite.embedded_effects_preserved
  ROOT_MIRROR_REQUIRED = $prewrite.root_mirror_required
  ROOT_MIRROR_SYNCED = $writeback.root_mirror_synced
  WRITE_SUCCESS = if ($writeSummary) { Read-JsonValue -Object $writeSummary -Keys @("WRITE_SUCCESS", "write_success") } else { $false }
  ENCRYPT_SUCCESS = if ($writeSummary) { Read-JsonValue -Object $writeSummary -Keys @("ENCRYPT_SUCCESS", "encrypt_success") } else { $false }
  COMMIT_PERFORMED = if ($writeSummary) { Read-JsonValue -Object $writeSummary -Keys @("commit_performed") } else { $false }
  WRITEBACK_SUCCESS = if ($writeSummary) { Read-JsonValue -Object $writeSummary -Keys @("writeback_success") } else { $false }
  READY_FOR_USER_MANUAL_QC = if ($writeSummary) { Read-JsonValue -Object $writeSummary -Keys @("ready_for_user_manual_qc") } else { $false }
  POST_WRITE_ACTUAL_DRAFT_AUDIT_REQUIRED_ON_COMMIT = $summary.post_write_actual_draft_audit_required_on_commit
  POST_WRITE_ACTUAL_DRAFT_AUDIT_EXECUTED = $summary.post_write_actual_draft_audit_executed
  POST_WRITE_ACTUAL_DRAFT_AUDIT_GATE_PASSED = $summary.post_write_actual_draft_audit_gate_passed
  POST_WRITE_ACTUAL_DRAFT_AUDIT_BLOCKER_CODES = (($summary.post_write_actual_draft_audit_blocker_codes | ForEach-Object { [string]$_ }) -join ",")
  ACTUAL_TEXT_RESIDUE_GATE_PASSED = $auditSummary.actual_text_residue_gate_passed
  ACTUAL_AUDIO_COVERAGE_GATE_PASSED = $auditSummary.actual_audio_coverage_gate_passed
  ACTUAL_VISIBLE_TEXT_REPEAT_GATE_PASSED = $auditSummary.actual_visible_text_repeat_gate_passed
  ACTUAL_HAS_NO_EXTRA_CAPTION_LIKE_TEXT_SEGMENTS = $auditSummary.actual_has_no_extra_caption_like_text_segments
  ACTUAL_CAPTION_ROWS_EXACT_MATCH_PLAN = $auditSummary.actual_caption_rows_exact_match_plan
  ACTUAL_AUDIO_COVERAGE_FAILURE_COUNT = $auditSummary.actual_audio_coverage_failure_count
  HEARD_BUT_UNCAPTIONED_WORD_COUNT = $auditSummary.heard_but_uncaptioned_word_count
  DROPPED_BUT_REINTRODUCED_WORD_COUNT = $auditSummary.dropped_but_reintroduced_word_count
  OLD_SUBTITLE_RESIDUE_COUNT = $auditSummary.old_subtitle_residue_count
  ORPHAN_TEXT_SEGMENT_COUNT = $auditSummary.orphan_text_segment_count
  TEXT_AFTER_FINAL_VIDEO_END_COUNT = $auditSummary.text_after_final_video_end_count
  BLOCKER_CODES = (($summary.blocker_codes | ForEach-Object { [string]$_ }) -join ",")
}

foreach ($item in $summaryLines.GetEnumerator()) {
  Write-Output ("{0}={1}" -f $item.Key, $item.Value)
}

if (-not $Commit) {
  Write-Output "DRY_RUN_PASS=true"
  Write-Output "WRITE_SKIPPED=true"
}
