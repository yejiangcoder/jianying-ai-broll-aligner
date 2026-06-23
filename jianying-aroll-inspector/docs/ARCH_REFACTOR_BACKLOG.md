# A-Roll V21 Architecture Refactor Backlog

This backlog records Wave 2 refactor targets only. These items are not part of
the current behavior-preserving Wave 1 implementation.

## Non-Negotiables

- Keep every refactor behavior-preserving unless the commander explicitly starts
  a quality-rule iteration.
- Do not add video sample text, draft names, or one-off production phrases to
  source code.
- Do not change real Jianying drafts while doing architecture work.
- Keep old public import paths compatible during each split.
- Run `py -3 -m compileall -q src tests tools`, `py -3 -m pytest -q`, and
  `git diff --check` after every work order.

## Work Order 5: Semantic Decision Planner Split

Target structure:

```text
src/aroll_v21/decision/
  semantic_decision_planner.py
  local_policy.py
  unit_split_binding.py
  semantic_json_planner.py
  deepseek_request_builder.py
  decision_trace.py
```

Scope:

- Move local deterministic policy wiring into `local_policy.py`.
- Move unit split binding helpers into `unit_split_binding.py`.
- Move JSON decision planner loading/parsing into `semantic_json_planner.py`.
- Move DeepSeek request shaping into `deepseek_request_builder.py`.
- Move trace row construction into `decision_trace.py`.

Do not change semantic adjudication decisions, blocker codes, request schema, or
DeepSeek provider behavior in this work order.

## Work Order 6: Visual Pacing Split

Target structure:

```text
src/aroll_v21/quality/visual_pacing/
  normalizer.py
  report.py
  merge_safety.py
  suffix_cleanup.py
  short_segment_padding.py
  cut_density.py
```

Scope:

- Keep `visual_pacing.py` as a compatibility facade while helpers move out.
- Move merge safety checks into `merge_safety.py`.
- Move suffix cleanup helpers into `suffix_cleanup.py`.
- Move short segment padding helpers into `short_segment_padding.py`.
- Move cut density metrics into `cut_density.py`.
- Move report assembly into `report.py`.

Do not alter merge thresholds, short segment thresholds, cut density thresholds,
or blocker semantics in this work order.

## Work Order 7: Final Timeline Compiler Split

Target structure:

```text
src/aroll_v21/compiler/
  final_timeline_compiler.py
  segment_builder.py
  unit_split_materializer.py
  timeline_repack.py
  compiler_report.py
```

Scope:

- Keep the current compiler import path compatible.
- Move segment construction into `segment_builder.py`.
- Move unit split materialization into `unit_split_materializer.py`.
- Move timeline repacking into `timeline_repack.py`.
- Move compiler report assembly into `compiler_report.py`.

Do not change segment ordering, target times, source times, word IDs, caption
binding, or report fields in this work order.
