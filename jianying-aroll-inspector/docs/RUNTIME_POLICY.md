# Runtime Policy

Runtime data is stored outside the source directory by default.

Default layout:

```text
D:\auto_clip_runtime\
  arll\
    runs\
    reports\
    backups\
    temp\
    cache\
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
  packages\
    release\
    dev_snapshot\
```

Rules:

- Do not create a junction or symlink from project `runtime/` to the external runtime.
- Do not add `D:\auto_clip_runtime` as an IDEA content root.
- Do not include runtime data in release packages or dev snapshots.
- Migration is dry-run by default.
- No source files are deleted during migration.
- Real Jianying draft folders are never moved by this tool.
