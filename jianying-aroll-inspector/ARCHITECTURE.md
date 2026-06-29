# A-Roll V21 Architecture Baseline

Status: `PHASE_0_BASELINE`

Date: 2026-06-29

This document freezes the current architecture shape before the next refactor
wave. It is a governance baseline, not a feature specification.

## Active Entry

The active V21 production entry is:

```text
run_aroll_v21_operator.ps1
-> src/aroll_v21.cli
-> src/aroll_v21.operator.run_operator
-> src/aroll_v21.engine.ArollEngine.run
```

The operator owns runtime boundary checks, real draft ingest, dry-run/write
mode selection, semantic provider construction, artifact writing, and writeback
commit orchestration.

`ArollEngine.run` owns the in-memory editing plan:

```text
ingest/source graph
-> repeat evidence
-> semantic decision plan
-> final timeline compile
-> quality mutation passes
-> captions
-> material write plan
-> read-only validators
-> blocker/report summary
```

## Current Quality Chain

The current quality chain is behaviorally stable but still too centralized.

Implemented layers:

- `ReadOnlyValidators` builds quality gates without mutating timeline, captions,
  or material write plans.
- `final_caption_visible` has explicit detector, classifier, policy, repair
  signal, gate, and semantic arbitration surfaces.
- `final_visible_repair` has an explicit context and transaction pipeline.
- Quality mutations record before/after signatures and can reject regressions.
- DeepSeek and other semantic providers are limited to semantic decisions.
- Final-visible repeat provider output is advisory only and does not mutate the
  final timeline.

Remaining centralization:

- `src/aroll_v21/engine.py` is still the largest orchestration object.
- `ArollEngine.run` still coordinates quality pass order, rechecks, cycle
  detection, validator attachment, semantic request merge, and write gates.
- `src/aroll_v21/quality/final_visible_caption_repair.py` still combines entry,
  dispatcher, aggregation, and historical repair glue.
- `src/aroll_v21/quality/final_caption_visible_repeat.py` still contains many
  detector families in one module.
- `quality_gate.py` and `engine_summary.py` still act as broad report-field
  buses.

## Non-Negotiable Boundaries

- Do not import or revive V20 patch orchestration.
- Do not add symptom-specific production phrases to source code.
- Do not let validators repair.
- Do not let writeback repair.
- Do not let writer fallback create captions.
- Do not let DeepSeek return or apply physical edit fields.
- Do not mutate real Jianying drafts during architecture work.
- Do not add another direct final-timeline rewrite path outside a recorded
  quality mutation or repair transaction.

## Timeline Mutation Policy

Every pass that can change `final_timeline` or `captions` must expose:

- input timeline/caption signature
- output timeline/caption signature
- action rows
- accepted/rejected state
- rejection reason
- whether a downstream quality recheck is required

The next refactor wave should move scattered quality-pass sequencing out of
`ArollEngine.run` and into a dedicated `QualityPipeline`.

## Semantic Provider Policy

Semantic providers may classify or adjudicate semantic ambiguity. They must not
directly decide physical edit boundaries.

Final-visible repeat advisory fields currently include:

- `final_visible_repeat_advisory_count`
- `final_visible_repeat_advisory_result_count`
- `final_visible_repeat_advisory_decision_counts`
- `final_visible_repeat_advisory_keep_count`
- `final_visible_repeat_advisory_drop_candidate_count`
- `final_visible_repeat_advisory_review_count`
- `final_visible_repeat_advisory_unresolved_count`
- `final_visible_repeat_advisory_applied_count`
- `final_visible_repeat_advisory_provider_called_count`
- `final_visible_repeat_advisory_policy`

The required policy value is:

```text
advisory_only_no_timeline_mutation
```

## Phase 0 Dirty Baseline

Before this document was added, the behavior changes from the latest quality
architecture phases were local and uncommitted.

Tracked files modified:

```text
src/aroll_v21/engine.py
src/aroll_v21/engine_summary.py
src/aroll_v21/quality/boundary_overlap.py
src/aroll_v21/quality/final_caption_visible/__init__.py
src/aroll_v21/quality/final_caption_visible/gate.py
src/aroll_v21/quality/final_caption_visible_repeat.py
src/aroll_v21/quality/final_visible_repeat_classification.py
src/aroll_v21/quality/quality_gate.py
tests/test_aroll_v21_quality_gates.py
tests/test_aroll_v21_repeat_gate_classification.py
tests/test_aroll_v21_semantic_request_consistency_gate.py
```

Untracked implementation file:

```text
src/aroll_v21/quality/final_caption_visible/semantic_arbitration.py
```

## Verification Baseline

Phase 0 verification commands:

```powershell
python -m pytest tests/test_aroll_v21_no_architecture_drift.py tests/test_aroll_v21_no_drift.py tests/test_aroll_v21_no_drift_allowlist.py tests/test_aroll_v21_no_v20_patch_imports.py -q
python -m pytest -q
git diff --check
```

Expected result:

- architecture drift tests pass
- full test suite passes
- no whitespace errors
- runtime-path warnings are acceptable when runtime env vars are unset

## Next Refactor Order

1. Split `ArollEngine.run` into explicit stage orchestration.
2. Move quality sequencing into `QualityPipeline`.
3. Complete final-visible repair dispatcher cleanup.
4. Split final-visible repeat detector families.
5. Replace broad report dictionaries with registered report schemas.
6. Add Python project tooling and refresh stale docs.
7. Build a durable quality case fixture library.

## Phase 1 Progress

Status: `COMPLETE`

`ArollEngine.run` is now a stage orchestration function instead of the full
pipeline body. It delegates to:

```text
_run_ingest_stage
_run_decision_stage
_run_compile_stage
_run_quality_stage
_run_writer_stage
_run_validation_stage
_build_final_run_report
```

The refactor is behavior-preserving. It does not introduce new quality rules,
change edit decisions, or write real drafts.

Known remaining stage-2 target:

- `_run_quality_stage` still owns the large quality-pass sequence. It should be
  split into a dedicated `QualityPipeline` next.

Verification:

```text
python -m compileall -q src tests tools
python -m pytest tests/test_aroll_v21_no_architecture_drift.py tests/test_aroll_v21_no_drift.py tests/test_aroll_v21_no_drift_allowlist.py tests/test_aroll_v21_no_v20_patch_imports.py tests/test_aroll_v21_static_hidden_bug_scan.py -q
python -m pytest tests/test_aroll_v21_semantic_request_consistency_gate.py tests/test_aroll_v21_quality_gates.py tests/test_aroll_v21_final_backend_integration_contract.py tests/test_aroll_v21_final_failure_path_matrix.py -q
python -m pytest -q
git diff --check
```

## Phase 2 Progress

Status: `COMPLETE`

The quality-pass sequence has moved out of `ArollEngine` and into:

```text
src/aroll_v21/quality/pipeline.py
```

The new module owns:

- `QualityPipelineHooks`
- `QualityPipelineResult`
- `QualityPipeline`
- `QualityPipeline.run`

`ArollEngine._run_quality_stage` now only wires dependencies into
`QualityPipelineHooks` and delegates to `QualityPipeline.run`. The old
`aroll_v21.engine.repair_final_visible_caption_issues` import is intentionally
kept as a compatibility injection point for existing tests and future controlled
repair substitution; the quality sequence itself remains in the pipeline.

Behavior boundary:

- no new quality rules
- no edit-decision changes
- no real draft writes
- no validator or writer repair
- no DeepSeek physical edit application

Architecture guard:

- `tests/test_aroll_v21_no_architecture_drift.py` now asserts that quality
  sequencing stays in `quality/pipeline.py` and does not drift back into
  `engine.py`.

Verification:

```text
python -m compileall -q src tests tools
python -m pytest tests/test_aroll_v21_no_architecture_drift.py tests/test_aroll_v21_no_drift.py tests/test_aroll_v21_no_drift_allowlist.py tests/test_aroll_v21_no_v20_patch_imports.py tests/test_aroll_v21_static_hidden_bug_scan.py -q
python -m pytest tests/test_aroll_v21_semantic_request_consistency_gate.py tests/test_aroll_v21_quality_gates.py tests/test_aroll_v21_final_backend_integration_contract.py tests/test_aroll_v21_final_failure_path_matrix.py -q
python -m pytest -q
git diff --check
```

Known remaining stage-3 target:

- `final_visible_caption_repair.py` still combines entry, dispatcher,
  aggregation, and historical repair glue. It should be reduced to a thin
  dispatcher over the existing final-visible repair transaction pipeline.

## Phase 3 Progress

Status: `COMPLETE`

The final-visible repair entry has been reduced by moving dispatcher-owned
state and rule registration into dedicated modules:

```text
src/aroll_v21/quality/final_visible_repair/loop_state.py
src/aroll_v21/quality/final_visible_repair/registry.py
```

`loop_state.py` owns:

- current timeline/caption/signature state
- seen-signature tracking
- accepted action accumulation
- unresolved repair rows
- pipeline-result consumption

`registry.py` owns:

- `FinalVisibleRepairRuleCallbacks`
- `FinalVisibleRepairRuleRegistry`
- deterministic transaction rule ordering
- proposal rule ordering
- open-tail and residual rule groups
- caption-only finalizer rule groups
- gate-candidate repair rule construction

`final_visible_caption_repair.py` still owns the public repair entry and the
historical repair helper functions, but it no longer embeds the rule-list
construction or the pipeline-result state machine.

Behavior boundary:

- no new quality rules
- no sample-specific phrase handling
- no rule ordering changes
- no real draft writes
- no validator or writer repair

Architecture guard:

- `tests/test_aroll_v21_no_architecture_drift.py` now asserts that final-visible
  repair rules are registered through `registry.py`, loop mutation is consumed
  through `loop_state.py`, and old `globals().update` / rule dependency
  configuration does not return.

Verification:

```text
python -m py_compile src\aroll_v21\quality\final_visible_caption_repair.py src\aroll_v21\quality\final_visible_repair\registry.py src\aroll_v21\quality\final_visible_repair\loop_state.py tests\test_aroll_v21_no_architecture_drift.py
python -m pytest tests/test_aroll_v21_no_architecture_drift.py tests/test_aroll_v21_no_drift.py tests/test_aroll_v21_no_drift_allowlist.py tests/test_aroll_v21_static_hidden_bug_scan.py -q
python -m pytest tests/test_aroll_v21_quality_gates.py tests/test_aroll_v21_final_visible_generic_qc_regressions.py tests/test_aroll_v21_jimei_qc_regressions_round12.py tests/test_aroll_v21_repeat_gate_classification.py -q
```

Known remaining stage-4 target:

- `final_visible_caption_repair.py` still contains many historical helper
  aliases and local repair functions. The next cleanup should move one helper
  family at a time into owned rule modules while keeping the registry order
  unchanged.

## Phase 4 Progress

Status: `COMPLETE`

The final-visible proposal materialization layer has moved out of
`final_visible_caption_repair.py` and into:

```text
src/aroll_v21/quality/final_visible_repair/proposal_apply.py
```

The new module owns:

- render callback adaptation for timeline proposal materialization
- proposal materialization into `_RepairStep`
- unresolved proposal rows
- proposal action rows and coverage summaries
- caption span-drop proposal construction
- boundary restart proposal repair application
- repeated island proposal repair application

The repair entry still owns the public orchestration and the remaining
historical helper families, but it no longer embeds boundary/repeated proposal
apply wrappers or duplicated span-drop materialization/report construction.

Behavior boundary:

- no detector or classifier condition changes
- no proposal ordering changes
- no action/report field shape changes
- no sample-specific phrase handling
- no real draft writes

Architecture guard:

- `tests/test_aroll_v21_no_architecture_drift.py` now asserts that proposal
  apply helpers live in `proposal_apply.py` and do not drift back into
  `final_visible_caption_repair.py`.

Verification:

```text
python -m py_compile src\aroll_v21\quality\final_visible_caption_repair.py src\aroll_v21\quality\final_visible_repair\proposal_apply.py tests\test_aroll_v21_no_architecture_drift.py
python -m pytest tests/test_aroll_v21_no_architecture_drift.py tests/test_aroll_v21_boundary_restart_repair.py tests/test_aroll_v21_repeated_island_repair.py tests/test_aroll_v21_quality_integration_gate.py -q
python -m pytest tests/test_aroll_v21_quality_gates.py::ArollV21QualityGateTests::test_final_visible_repair_keeps_progressive_semantic_expansion tests/test_aroll_v21_quality_gates.py::ArollV21QualityGateTests::test_final_visible_repair_trims_repeated_short_discourse_opener -q
```

Known remaining stage-5 target:

- Move the next isolated helper family out of `final_visible_caption_repair.py`,
  preferably open-tail / short-aborted caption proposal helpers, while keeping
  the registry order unchanged.

## Phase 5 Progress

Status: `COMPLETE`

The short caption fragment repair family has moved out of
`final_visible_caption_repair.py` and into:

```text
src/aroll_v21/quality/final_visible_repair/rules/caption_fragment.py
```

The new module owns:

- contained short fragment proposal repair
- self-repair aborted phrase proposal repair
- short-aborted-prefix caption proposal repair
- open-tail short caption merge repair
- short-aborted prefix candidate classification
- open-tail merge eligibility checks
- contained-fragment drop selection
- the family-specific constants for short/open-tail caption handling

The repair entry still wires this family through `FinalVisibleRepairRuleRegistry`
callbacks. Open-tail repair receives
`render_captions_preserving_caption_only_materializations` as an explicit
callback so the existing caption-only materialization behavior is preserved
without importing the public entry module from a rule module.

Behavior boundary:

- no detector or classifier condition changes
- no proposal ordering changes
- no caption render behavior changes
- no action/report field shape changes
- no sample-specific phrase handling
- no real draft writes

Architecture guard:

- `tests/test_aroll_v21_no_architecture_drift.py` now asserts that the caption
  fragment family lives in `rules/caption_fragment.py`, that the repair entry
  only wires it through `_caption_fragment_rules`, and that the old local helper
  functions/constants do not drift back into `final_visible_caption_repair.py`.

Verification:

```text
python -m py_compile src\aroll_v21\quality\final_visible_caption_repair.py src\aroll_v21\quality\final_visible_repair\rules\caption_fragment.py tests\test_aroll_v21_no_architecture_drift.py
python -m pytest tests/test_aroll_v21_no_architecture_drift.py tests/test_aroll_v21_static_hidden_bug_scan.py -q
python -m pytest tests/test_aroll_v21_quality_integration_gate.py tests/test_aroll_v21_quality_gates.py::ArollV21QualityGateTests::test_final_visible_repair_trims_repeated_short_discourse_opener tests/test_aroll_v21_quality_gates.py::ArollV21QualityGateTests::test_final_visible_repair_trims_repeated_short_discourse_opener_inside_merged_segment tests/test_aroll_v21_quality_gates.py::ArollV21QualityGateTests::test_final_visible_repair_keeps_restart_lead_after_internal_prefix_fragment -q
```

Known remaining stage-6 target:

- Move the next bounded helper family out of `final_visible_caption_repair.py`,
  likely fatal tiny caption proposal repair or caption-level final-repeat
  aborted containment, while preserving the current registry order.

## Phase 6 Progress

Status: `COMPLETE`

Fatal tiny caption proposal repair has moved out of
`final_visible_caption_repair.py` and into the caption fragment rule family:

```text
src/aroll_v21/quality/final_visible_repair/rules/caption_fragment.py
```

The caption fragment module now owns:

- fatal tiny caption classification lookup
- tiny-caption residual proposal construction
- tiny-caption residual proposal application through `proposal_apply.py`
- the existing contained-fragment, self-repair-aborted, short-aborted-prefix,
  and open-tail short-caption repairs from phase 5

The repair entry still wires the callback through `FinalVisibleRepairRuleRegistry`
but no longer imports `TimelineRepairProposal`, `build_tiny_caption_classification_report`,
or a local tiny-residual proposal apply helper.

Behavior boundary:

- no tiny-caption classifier changes
- no proposal ordering changes
- no action/report field shape changes
- no sample-specific phrase handling
- no real draft writes

Architecture guard:

- `tests/test_aroll_v21_no_architecture_drift.py` now asserts that fatal tiny
  caption repair lives in `rules/caption_fragment.py`, that the repair entry
  only wires it through `_caption_fragment_rules`, and that the old local
  implementation/import shape does not drift back into
  `final_visible_caption_repair.py`.

Verification:

```text
python -m py_compile src\aroll_v21\quality\final_visible_caption_repair.py src\aroll_v21\quality\final_visible_repair\rules\caption_fragment.py tests\test_aroll_v21_no_architecture_drift.py
python -m pytest tests/test_aroll_v21_no_architecture_drift.py tests/test_aroll_v21_static_hidden_bug_scan.py -q
python -m pytest tests/test_aroll_v21_quality_integration_gate.py::test_final_visible_repairs_fatal_tiny_caption_residual_through_proposal tests/test_aroll_v21_boundary_restart_repair.py::test_boundary_restart_repair_trims_previous_suffix_and_keeps_next_complete -q
```

Known remaining stage-7 target:

- Move caption-level final-repeat aborted containment helpers into a dedicated
  final-repeat caption repair module, keeping `_repair_next_issue` and registry
  ordering unchanged.

## Phase 7 Progress

Status: `COMPLETE`

Caption-level final-repeat aborted containment repair has moved out of
`final_visible_caption_repair.py` and into:

```text
src/aroll_v21/quality/final_visible_repair/rules/final_repeat_caption.py
```

The new module owns:

- caption rows used by `build_final_repeat_gate_report`
- aborted containment drop-caption selection
- relaxed containment matching
- final-target-repeat caption containment repair dispatch

The repair entry still wires the callback through `FinalVisibleRepairRuleRegistry`
but no longer imports `build_final_repeat_gate_report` or contains the
caption-level final-repeat helper functions.

Behavior boundary:

- no final-repeat detector changes
- no `_repair_next_issue` changes
- no registry ordering changes
- no action/report field shape changes
- no sample-specific phrase handling
- no real draft writes

Architecture guard:

- `tests/test_aroll_v21_no_architecture_drift.py` now asserts that
  caption-level final-repeat containment logic lives in
  `rules/final_repeat_caption.py`, that the repair entry only wires it through
  `_final_repeat_caption_rules`, and that the old local helper functions/imports
  do not drift back into `final_visible_caption_repair.py`.

Verification:

```text
python -m py_compile src\aroll_v21\quality\final_visible_caption_repair.py src\aroll_v21\quality\final_visible_repair\rules\final_repeat_caption.py tests\test_aroll_v21_no_architecture_drift.py
python -m pytest tests/test_aroll_v21_no_architecture_drift.py tests/test_aroll_v21_static_hidden_bug_scan.py -q
python -m pytest tests/test_aroll_v21_quality_gates.py::ArollV21QualityGateTests::test_final_visible_repair_drops_caption_level_aborted_final_repeat_containment -q
```

Known remaining stage-8 target:

- Continue reducing `final_visible_caption_repair.py` by moving the next
  bounded helper family, likely semantic-integrity repair or gate-candidate
  dispatch helpers, without changing `_repair_next_issue` ordering.
