# A-Roll Quality Defect Ledger

The defect ledger turns manual QC findings into structured evidence and regression-test work items.

Default output is external runtime storage, not the repo:

```text
D:\auto_clip_runtime\aroll_v21_audits\quality_defect_ledger
```

The directory is bound through runtime config:

```yaml
aroll:
  quality_defect_ledger_dir: "D:/auto_clip_runtime/aroll_v21_audits/quality_defect_ledger"
```

Environment override:

```powershell
$env:AUTO_CLIP_AROLL_QUALITY_DEFECT_LEDGER_DIR = "D:\auto_clip_runtime\aroll_v21_audits\quality_defect_ledger"
```

## One Issue

```powershell
py -3 tools\aroll_quality_defect_ledger.py `
  --run-dir "D:\auto_clip_runtime\aroll_v21_uat_runs\<run>" `
  --case-id "qc_20260619_jimei_round1" `
  --draft-label "6月19日" `
  --issue-text "你真的以为你在评论/区里面敲了几个字" `
  --root-cause "cross_caption_semantic_containment" `
  --note "manual QC: visible caption should stay semantic whole"
```

## Multiple Issues

Create a private runtime JSON file, for example:

```json
{
  "issues": [
    {
      "issue_no": "1",
      "bad_visible_text": "visible text copied from QC",
      "root_cause": "restart_repeat_not_removed",
      "expected_visible_text": "optional expected caption",
      "note": "optional operator note"
    }
  ]
}
```

Then run:

```powershell
py -3 tools\aroll_quality_defect_ledger.py `
  --run-dir "D:\auto_clip_runtime\aroll_v21_uat_runs\<run>" `
  --issues-json "D:\auto_clip_runtime\aroll_v21_audits\qc_issues.json" `
  --case-id "qc_20260619_round1"
```

## Outputs

Each case writes:

```text
defect_ledger.json
defect_ledger.md
```

The ledger includes:

- matched caption ids and neighboring visible captions
- final timeline segment ids, source media, source times, word ids, native words
- whether the issue entered semantic request payloads, DeepSeek decisions, final visible repair, final caption repeat gate, and quality gate
- why the existing gates likely passed
- suggested generic regression test file, test name, mechanism, and assertion

## Root Cause Labels

Preferred labels:

```text
dangling_prefix_or_suffix
caption_boundary_split_error
semantic_garbage_caption
asr_text_error
cross_caption_semantic_containment
restart_repeat_not_removed
```

If the issue does not fit, leave `root_cause` blank; the ledger will mark it `unclassified`, and the next code change should add a generic category instead of a sample-specific rule.
