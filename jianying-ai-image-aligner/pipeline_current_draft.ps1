function Resolve-ImageAlignerDraftDir {
  param(
    [string]$DraftDir
  )

  if (![string]::IsNullOrWhiteSpace($DraftDir)) {
    return $DraftDir
  }

  $StatePath = $env:VIDEO_PIPELINE_CURRENT_DRAFT_STATE
  if ([string]::IsNullOrWhiteSpace($StatePath)) {
    $StatePath = "D:\auto_clip_runtime\video_pipeline\current_draft.json"
  }
  if (!(Test-Path -LiteralPath $StatePath)) {
    throw "DraftDir is empty and current draft state does not exist: $StatePath. Run run_bind_current_draft.ps1 after A-Roll QC passes."
  }

  $State = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
  if ([string]$State.version -ne "video_pipeline_current_draft_v1") {
    throw "Current draft state version is invalid: $StatePath"
  }
  if ($State.aroll_qc_passed -isnot [bool] -or $State.aroll_qc_passed -ne $true) {
    throw "Current draft state exists but A-Roll QC is not marked passed: $StatePath"
  }
  $ResolvedDraftDir = [string]$State.draft_dir
  if ([string]::IsNullOrWhiteSpace($ResolvedDraftDir)) {
    throw "Current draft state has no draft_dir: $StatePath"
  }
  if (!(Test-Path -LiteralPath $ResolvedDraftDir)) {
    throw "Current draft dir from state does not exist: $ResolvedDraftDir"
  }
  if ([string]::IsNullOrWhiteSpace([string]$State.timeline_id)) {
    throw "Current draft state has no timeline_id; re-run run_bind_current_draft.ps1 after A-Roll QC passes: $StatePath"
  }

  Write-Host "AUTO_BOUND_DRAFT_STATE=$StatePath"
  Write-Host "AUTO_BOUND_DRAFT_DIR=$ResolvedDraftDir"
  return $ResolvedDraftDir
}
