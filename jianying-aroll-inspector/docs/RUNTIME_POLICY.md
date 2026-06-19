# Runtime Policy

Runtime data is stored outside the source directory by default.

Default layout:

```text
%USERPROFILE%\.auto_clip_runtime\
  aroll_v21_uat_runs\
  aroll_v21_audits\
  aroll_v21_test_outputs\
  aroll_v21_backups\
  broll\
    design_runs\
    material_index\
    downloaded_materials\
  ai_images\
    batches\
    manifests\
  drafts\
    real_drafts\
    draft_backups\
  exports\
  logs\
  temp\
  cache\
  packages\
    release\
    dev_snapshot\
```

Rules:

- Do not create a junction or symlink from project `runtime/` to the external runtime.
- Do not add `%USERPROFILE%\.auto_clip_runtime` as an IDEA content root.
- Do not include runtime data in release packages or dev snapshots.
- Migration is dry-run by default.
- No source files are deleted during migration.
- Real Jianying draft folders are never moved by this tool.
