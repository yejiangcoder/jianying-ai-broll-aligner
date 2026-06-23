# Runtime Policy

`jianying-ai-image-aligner` stores runtime output outside the project root.

Default runtime root:

```text
D:\auto_clip_runtime\image_aligner
```

Default run output:

```text
D:\auto_clip_runtime\image_aligner\runs
```

Rules:

- The project root should not contain `runtime/`.
- Do not create a symlink or junction from project `runtime/` back to external runtime.
- Do not add `D:\auto_clip_runtime` to IDEA as a content root.
- Keep `vendor/` in the project for now because it may be required by local execution.
- Codex should not scan `vendor/` unless the user explicitly asks.
- Generated logs and reports should go under the external runtime root.
