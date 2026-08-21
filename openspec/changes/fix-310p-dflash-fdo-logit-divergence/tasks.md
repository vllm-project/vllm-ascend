## 1. Freeze and reproduce the diagnostic baseline

- [x] 1.1 Record the plugin branch, HEAD, staged/unstaged inventory, imported plugin path, frozen upstream vLLM HEAD/cleanliness, package versions, model paths, and NPU availability without modifying the existing staged FDO work.
- [x] 1.2 Extract the archived 35B TP2 C1 Eager/FDO token comparison and retain GSM8K record 12, first differing token index 48, launch configuration, and graph evidence as the diagnostic seed manifest.
- [x] 1.3 Start fresh probes-disabled Eager and FDO services for GSM8K record 12 at TP2, concurrency 1, output 64, temperature 0, ignore EOS, DFlash K=15, and capture sizes `[64,32,16]`.
- [x] 1.4 Repeat both modes and prove Eager determinism plus stable FDO reproduction before adding observability; if the branch no longer reproduces, stop and update the OpenSpec evidence instead of changing code.

Baseline evidence (2026-08-20): the stronger 16-request/output-256 paired run was used so the seed request and the full cluster were preserved together. Eager artifacts are under `/home/whn/aisbench_runs/fdo_logit_diag_baseline_20260820_eager1b` and `/home/whn/aisbench_runs/fdo_logit_diag_baseline_20260820_eager2`; only request 6 had a one-off Eager branch, while request 12 was token-identical. FDO artifacts are under `/home/whn/aisbench_runs/fdo_logit_diag_baseline_20260820_fdo1` and `/home/whn/aisbench_runs/fdo_logit_diag_baseline_20260820_fdo2b`; both reproduced eight Eager/FDO mismatches and request 12 at token 48, with complete target/draft rank0/rank1 capture and replay for descriptors 16/32/64 and zero safety errors. The discarded `/home/whn/aisbench_runs/fdo_logit_diag_baseline_20260820_fdo2` startup was a documented transient memory-reclamation failure (35.53 GiB free versus 36.75 GiB required) and was rerun unchanged after the cards returned idle.

## 2. Build the bounded trace schema and offline comparator with TDD

- [x] 2.1 Write failing tests for exact-scope probe configuration, including default-off behavior, explicit output directory, component/layer selection, iteration and byte bounds, and rejection outside 310P DFlash Eager/FDO.
- [x] 2.2 Implement the minimal immutable probe configuration and rerun the focused tests to green without allocating tensors or installing hooks when disabled.
- [x] 2.3 Write failing tests for a trace identity containing mode, component, TP rank, dataset request, generated prefix, speculative iteration, draft substep, descriptor, actual tokens, active-row mapping, semantic role, shape, and dtype.
- [x] 2.4 Implement bounded JSONL manifest records and owner-controlled tensor artifacts with atomic per-record completion; verify incomplete or over-bound artifacts fail closed.
- [x] 2.5 Write failing table-driven tests for active-lane alignment, padding exclusion, exact unequal counts, finite checks, max/mean absolute difference, relative difference, cosine similarity, top-k overlap, selected-token cross-logits, and argmax margins.
- [x] 2.6 Implement the offline paired comparator and verify it rejects mismatched prefixes, row mappings, ranks, components, descriptors, or truncated records instead of comparing unrelated tensors.
- [x] 2.7 Run the complete new trace/comparator unit set and existing graph-input-contract tests before integrating model hooks.

## 3. Add boundary probes through RED-GREEN cycles

- [x] 3.1 Write a failing target-runner test proving an enabled probe records active input IDs, positions, metadata identity, selected final hidden rows, logits/top-k evidence, and graph runtime identity after model execution.
- [x] 3.2 Implement the minimal target boundary integration and prove disabled execution performs no probe calls, CPU copies, filesystem work, or graph-routing changes.
- [x] 3.3 Write failing DFlash proposer tests proving matched draft iterations expose active inputs, remapped proposed token IDs, descriptor/runtime identity, and multi-request generated-prefix identity.
- [x] 3.4 Implement the minimal draft boundary integration without moving proposal/model/logits work outside the existing FULL wrapper.
- [x] 3.5 Stop the boundary ladder before adding sampler instrumentation after the first divergence is proven inside target/draft model execution; retain the existing sampler DFX and tests unchanged.
- [x] 3.6 Verify all numerical tracing remains an observer and never supplies a sampling or dispatch decision.
- [x] 3.7 Add tests for post-replay export, repeated descriptor reuse, TP-rank separation, active-lane slicing, artifact bounds, and runtime context-row slicing when Python hooks retain capture-time state.
- [x] 3.8 Run the focused target, proposer, rejection, ACL-graph, FDO contract, and acceptance-harness regression tests.

## 4. Add a replay-safe single-layer probe with TDD

- [x] 4.1 Write failing tests that resolve explicitly selected target or draft transformer layers, reject ambiguous/missing selections, and install no hooks for disabled services.
- [x] 4.2 Write failing lifecycle tests proving selected-layer storage is allocated before capture, retains its address, is updated by capture/replay, and is exported only after replay completion.
- [x] 4.3 Implement persistent device side-channel buffers and device-to-device copies that participate in the genuine graph while leaving normal execution unmodified.
- [x] 4.4 Add output normalization for tensor and tuple returns, strict shape/dtype/active-bound validation, dynamic GDN role widths/row multipliers, and per-rank artifact identity.
- [x] 4.5 Prove capture-time records cannot be mistaken for runtime replay and prove the added copies preserve target/draft FULL replay in focused hardware probes.

## 5. Extend the diagnostic runner

- [x] 5.1 Extend the frozen acceptance harness tests for source identity, exact flags, two-rank graph evidence, process-group ownership, port/card isolation, and guaranteed cleanup.
- [x] 5.2 Use bounded diagnostic launch scripts for fresh Eager/FDO services and preserve each server/trace artifact under a unique `/home/whn/aisbench_runs` root.
- [x] 5.3 Require formal runs to prove target/draft capture, replay, and native graph files on every TP rank; exclude all probe-enabled runs from performance reporting.
- [x] 5.4 Verify failed startup/comparison and successful runs clean only runner-owned processes and release their recorded ports/cards.

## 6. Locate the first divergent boundary on hardware

- [x] 6.1 Run repeated boundary-only paired diagnostics and align traces by generated prefix and graph iteration.
- [x] 6.2 Compare scheduler/model inputs first and prove input IDs, positions, active lengths, selected rows, and descriptor identity match before model-layer probing.
- [x] 6.3 Compare target and draft hidden/logit boundaries and locate the first transition inside model execution rather than the sampler.
- [x] 6.4 Preserve logits/top-k and active tensor artifacts together with component, rank, iteration, descriptor, and the preceding matching boundary.
- [x] 6.5 Update `design.md` and this task evidence with the proven draft context RoPE extent boundary.

## 7. Locate the first differing layer

- [x] 7.1 Enumerate the target's 32 transformer layers and verify Eager/FDO resolve the same ordered checkpoints.
- [x] 7.2 Run a bounded all-layer checkpoint sweep for matched batch-4 iterations and identify layer 1 as the first differing target layer.
- [x] 7.3 Preserve repeated matched-prefix artifacts and reject scheduler ticks whose input hash differs before comparing layers.
- [x] 7.4 Confirm layer 0 output and layer 1 input match while layer 1 output differs, then descend through input norm and linear-attention sub-boundaries.
- [x] 7.5 Locate the residual signature at layer-1 GDN core; MoE routing instrumentation is not applicable to this first differing 9B stage.
- [x] 7.6 Update the OpenSpec with the GDN-core signature and the Eager-to-Eager control magnitude.

## 8. Prove one root-cause hypothesis at a time

- [x] 8.1 State the context-extent hypothesis: FDO truncates draft context RoPE when the live context exceeds the query descriptor, predicting the first difference at context RoPE and downstream attention.
- [x] 8.2 Apply the single-variable experiment that prepares context rows through `max(query_descriptor, context_actual)` while leaving query graph shape unchanged.
- [x] 8.3 Compare adjacent context/query RoPE, attention, output projection, and layer-hidden boundaries; do not stack W8A8 or MoE experiments.
- [x] 8.4 Reproduce the correction in unit and hardware gates while preserving genuine target/draft FULL replay on both ranks.
- [x] 8.5 Confirm the correction is plugin-side Python and requires no upstream, operator, model-weight, or eager-island scope expansion.

## 9. Implement the minimal repair with TDD

- [x] 9.1 Write a focused regression test requiring a 64-token live context to remain intact with a 16-token query descriptor and confirm the old truncating behavior fails it.
- [x] 9.2 Implement only the context-descriptor correction and run the focused test to GREEN.
- [x] 9.3 Preserve the exact 310P DFlash FDO scope; Eager, Piecewise, other graph modes, non-DFlash, upstream, W8A8, MoE, and operators are unchanged.
- [x] 9.4 Run the probe/repair tests plus related runner, DFlash, rejection sampler, W8A8, GDN, ACL-graph, graph-contract, attention-mask, and Piecewise suites.

## 10. Pass fast hardware correctness gates

- [x] 10.1 With probes enabled, verify repaired context/query RoPE and adjacent attention/layer boundaries while genuine target/draft replay remains active on both ranks.
- [x] 10.2 With probes disabled, run 35B TP2 C1 records 0-15 at output 256; classify the Eager-repeat branch and retain only one isolated non-blocking FDO wording branch.
- [x] 10.3 Run 9B TP1 C1/C4/C10 gates and reject request, GatherV2/MTE/ACL/contract, acceptance, throughput, or graph-proof failures.
- [x] 10.4 Confirm probe artifacts remain bounded and probe-disabled startup/runtime allocates no probe buffers or filesystem artifacts.

## 11. Re-run the complete FDO acceptance matrix

- [x] 11.1 Run 9B TP1 C1/C4/C10, 9B TP2 C1/C4, and 35B TP2 C1/C4 on GSM8K records 0-15 with output 256 and the frozen Eager/FDO configuration.
- [x] 11.2 Require all requests to succeed, mean accepted length at least 5.0 and 90% of Eager, acceptance-rate loss no greater than 5 percentage points, and output throughput at least 85% of Eager.
- [x] 11.3 Require complete target/draft capture and runtime replay for every configured descriptor on every TP rank, expected NONE transitions, and zero unexpected fallback, contract, GatherV2, MTE, ACL, or traceback errors.
- [x] 11.4 Apply repeated Eager/FDO classification: unstable branch sets are numerical drift; one remaining stable isolated wording branch is recorded under the user-approved non-blocking policy.
- [x] 11.5 Record graph memory, KV-cache capacity, peak NPU memory, and probes-disabled performance; no unexplained threshold regression remains.

Apply evidence (2026-08-20): all seven FDO groups completed 16/16 requests. Output-throughput ratios were 1.135/1.010/1.023 for 9B TP1 C1/C4/C10, 1.605/1.170 for 9B TP2 C1/C4, and 2.191/1.630 for 35B TP2 C1/C4. FDO accepted lengths were 6.6230, 6.5864, 6.6957, 6.6115, 6.5204, 6.9698, and 6.7198 respectively. Every formal FDO summary contains the required target/draft descriptors on every TP rank and an empty `safety_errors` list. Exact artifact roots and branch classification are recorded in `design.md`.

## 12. Final verification and handoff

- [x] 12.1 Run the complete related unit suite (`272 passed`), unchanged Piecewise suite (`9 passed`), repository formatting/lint/static checks, `git diff --check`, and strict validation of both OpenSpec changes from a fresh shell.
- [x] 12.2 Verify upstream vLLM remains clean/frozen at `ee0da84ab9e04ac7610e28580af62c365e898389` and the plugin diff contains no C++, AscendC, custom-operator, generated log, model, credential, profiler, or unrelated source files.
- [x] 12.3 Update this change and the parent FDO evidence with root cause, RED-GREEN evidence, first-layer artifacts, formal results, residual limitations, and exact artifact paths.
- [x] 12.4 Clean runner-owned services, ports, cards, and temporary probe state while preserving bounded diagnostic and formal evidence directories.
- [x] 12.5 Present the complete diff and evidence for review before committing or pushing; retain the default-off DEBUG probe as a supported diagnostic capability. The final review found no Critical issue, and its scope/reproducibility findings were closed by exact FDO predicates plus the synchronized 4B/35B acceptance matrix.
