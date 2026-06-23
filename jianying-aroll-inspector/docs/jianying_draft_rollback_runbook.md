# Jianying Draft Rollback Runbook

Use this when a disposable Jianying draft must be returned to the clean baseline for the next UAT/QC pass.

Primary script:

```powershell
scripts/rollback_jianying_draft.ps1 `
  -DraftDir "<path-to-jianying-draft>" `
  -ExpectedSourceMediaRoot "<path-to-original-media-root>"
```

Without `-Apply`, the script is analyze-only. It decrypts backup candidates, selects the most likely clean baseline, verifies the selected baseline against `-ExpectedSourceMediaRoot` when provided, and writes a report under the configured runtime `draft_rollback_runs` directory.

To apply the rollback:

```powershell
scripts/rollback_jianying_draft.ps1 `
  -DraftDir "<path-to-jianying-draft>" `
  -ExpectedSourceMediaRoot "<path-to-original-media-root>" `
  -Apply `
  -StopJianying
```

Apply is intentionally fail-closed. It refuses to run without `-ExpectedSourceMediaRoot` unless `-Force` is explicitly provided. If the selected baseline confidence is `medium`, Apply also requires either an explicit `-BaselinePath` or `-Force`.

## Baseline Selection

The analyzer treats a clean baseline as a decryptable `draft_content` backup with:

- at least one video segment;
- no V21/A-Roll automation markers such as `v21_`, `aroll_v21`, `generated_caption`, or `source_segment_template`;
- a backup timestamp before the first dirty automation backup, when dirty backups are present.

Selection strategy:

1. Decrypt every `.bak` under `.backup`, including timeline-specific backup folders.
2. Classify candidates as clean-like or dirty-like.
3. If the draft already has a registered rollback baseline under the runtime `draft_rollback_runs\_rollback_baselines` directory, use that exact baseline.
4. Otherwise, if automation-dirty backups exist, select the latest clean-like backup before the first dirty-like backup. This preserves the user-prepared baseline: imported original video, 1.2x speed, beauty/effects, and Jianying's own saved state before V21 writes.
5. If no dirty-like backup exists, select the latest clean-like backup in the initial clean backup cluster. This avoids treating later manual edits, extra tracks, or other non-V21 changes as the original baseline when there is no automation marker.
6. Quarantine every backup entry later than the selected original baseline, not only automation-marked dirty entries.
7. If the automatic choice is not right, rerun with `-BaselinePath "<exact backup file>"`.
8. If confidence is `medium`, do not rely on automatic Apply; inspect the report and pass `-BaselinePath` explicitly.

The old strategy is still available for diagnostics:

```powershell
scripts/rollback_jianying_draft.ps1 -DraftDir "<path-to-jianying-draft>" -SelectionMode latest-clean-before-dirty
```

## Apply Behavior

When `-Apply` is used, the script:

- stops Jianying/CapCut processes only when `-StopJianying` is provided;
- preserves the current active draft files into the run quarantine directory;
- quarantines dirty backup entries and `timeline_backup_manifest.json` so Jianying cannot restore the processed draft as the latest backup;
- quarantines all backup entries after the selected original baseline so later manual track edits cannot be restored as the latest backup;
- copies the selected clean baseline into root and every active `Timelines\<id>` mirror for `draft_content.json`, `draft_content.json.bak`, and `template-2.tmp`;
- decrypts the active draft again and fails if any active target still contains automation markers or hash mismatches.

Because Jianying stores tracks and captions inside `draft_content`, extra tracks added later are removed by restoring all active `draft_content` mirrors to the selected original baseline. Extra media/resource files can remain on disk as harmless orphan files; they are not referenced after rollback.

The script does not run UAT, does not run V20/legacy code, and does not modify source code.
