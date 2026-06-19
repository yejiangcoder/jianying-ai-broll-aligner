# IDEA Setup

Open only:

```text
<repo>\jianying-aroll-inspector
```

Do not add this directory as a content root:

```text
%USERPROFILE%\.auto_clip_runtime
```

Mark these project-local directories as excluded if they still exist:

- `runtime/`
- `release/`
- `dev_snapshot/`
- `__pycache__/`
- `.pytest_cache/`

Recommended IDEA Codex scope:

- `src/`
- `tests/`
- `config/`
- `docs/`
- `tools/`
- `run_*.ps1`

Large tasks should run through Desktop Codex, not IDEA Codex.
