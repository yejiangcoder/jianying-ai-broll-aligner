# Project Layout

`<suite-root>\jianying-aroll-inspector` is the lightweight source directory.

Keep in this directory:

- `src/`
- `tests/`
- `profiles/`
- `config/`
- `docs/`
- `tools/`
- `run_*.ps1`
- `README.md`
- `README_UAT.md`
- `AGENTS.md`
- `.env.example`
- `.gitignore`

Keep outside this directory:

- runtime runs
- generated reports
- draft backups
- release packages
- dev snapshots
- temporary audio
- media files
- decrypted or encrypted draft JSON artifacts

Default external runtime root:

```text
%USERPROFILE%\.auto_clip_runtime
```
