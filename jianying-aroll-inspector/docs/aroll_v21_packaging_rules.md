# A-Roll V21 Packaging Rules

Use `tools/package_aroll_v21_audit_zip.py` for architecture audit handoff packages.

Include:

- `src/`
- `tests/`
- `tools/`
- `docs/`
- `fixtures/`
- `run_aroll_v21_operator.ps1`
- README / requirements / project config files when present

Exclude:

- `.git/`
- `.idea/`
- `.vscode/`
- `.pytest_cache/`
- `__pycache__/`
- `*.pyc`
- virtual environments
- `runtime/`
- ad hoc `run_dir/`
- real Jianying drafts
- media files
- `.env`
- secrets / API keys
- `draft_content.json`
- `template-2.tmp`

Example:

```powershell
py -3 tools/package_aroll_v21_audit_zip.py `
  --output-zip D:\auto_clip_runtime\aroll_v21_audit.zip
```

The packaging script is deterministic enough for audit handoff but does not run real UAT, write drafts, or encrypt/decrypt content.
