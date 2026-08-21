## 1. Freeze the baseline and reproduce the first current failure

- [x] 1.1 Record the plugin commit (`959b9a6a`), upstream vLLM commit, installed package versions, CANN/driver/firmware versions, model paths, visible devices, and exact launch commands in the acceptance evidence directory.
- [x] 1.2 Confirm upstream vLLM is clean and establish a fresh 9B TP1 eager control with the frozen DFlash flags before changing code.
- [x] 1.3 Launch a fresh 9B TP1 `FULL_DECODE_ONLY` control with capture sizes `[64,32,16]`, preserve the complete startup/request traceback, and identify the first current failure rather than assuming a historical failure is still primary.
- [x] 1.4 Classify that first failure against the agreed boundaries: plugin Python fix only, no upstream vLLM change, no C++/AscendC/custom-operator edit, and no hidden eager island.
- [x] 1.5 Archive prior reverted commits and historical `aclnnCausalConv1dV310`/alignment logs as evidence only; document explicitly which ideas, if any, remain independently justified.

## 2. Implement and test the exact-scope execution policy

- [x] 2.1 Add failing unit tests for an exact activation predicate requiring 310P, DFlash speculative decoding, and `FULL_DECODE_ONLY` simultaneously.
- [x] 2.2 Add negative tests proving the predicate does not alter eager, `PIECEWISE`, `FULL`, `FULL_AND_PIECEWISE`, non-DFlash, non-speculative, or non-310P execution.
- [x] 2.3 Add a closed execution-state classifier covering prefill, chunked prefill, prefix-cache transition, mixed prefill/decode, and uniform DFlash decode.
- [x] 2.4 Add descriptor-arithmetic tests proving K=15 produces decode query length 16 and maps valid request batches to capture descriptors 16, 32, and 64 without confusing token count with sequence count.
- [x] 2.5 Implement the minimal exact-scope predicate and closed classifier in the plugin, with prefill/transition/mixed states selecting `NONE` and eligible uniform decode selecting `FULL`.
- [x] 2.6 Integrate the classifier into graph dispatch without globally forcing `FULL` and without changing existing Piecewise dispatch.
- [x] 2.7 Add tests for classified eager fallback: INFO may report an unexpected non-safety fallback, while DEBUG and acceptance mode fail an eligible-decode fallback.

- [x] 2.8 Freeze the current regressed concurrency-10 diff, launch commands, 9B C1/C4/C10 logs, the historical healthy table, and the fresh 3.28-3.30 acceptance evidence without treating the regressed state as a baseline.
- [x] 2.9 Reverse only the concurrency-10 position-tail and draft-RoPE delta, preserve the base FDO implementation and unrelated changes, and prove 9B TP1 C1/C4 recover under the 256-token workload.
- [x] 2.10 Reproduce the original 9B TP1 C10 failure from the recovered state and retain its first causal error, active request count, descriptor transition, and graph component.
- [x] 2.11 Add failing focused tests and default-off DEBUG evidence that compare active-lane positions, RoPE inputs, first differing draft token/probability, and update ordering while permitting initialized inactive tails.
- [x] 2.12 Test one root-cause hypothesis at a time, beginning at the first proven divergence; keep the earlier padding, staging, synchronization, and pointer experiments as evidence rather than stacking them into a candidate.
- [x] 2.13 Implement the smallest plugin-side fix that closes the proven C10 root cause without moving required model work into an eager island or modifying upstream/operator code.
- [x] 2.14 After each candidate, run the four-prompt 9B TP1 C1/C10 fast gate with output 256 and reject any request failure, output mismatch, graph error, or acceptance length below 90% of the recovered control.
- [x] 2.15 After the fast gate passes, run 9B TP1 GSM8K 0-15 at C1/C4/C10 with output 256 and require 16/16 success, the approved deterministic-output classification policy, acceptance and throughput thresholds, and genuine target/draft FULL replay evidence.
- [x] 2.16 Preserve the passing evidence and resume TP2/35B work only after every 9B gate passes.

Current C10 root-cause evidence (2026-08-19): the recovered implementation fails only after a ten-request descriptor of 160 becomes sparse (`actual=96`, later `80`) while replay retains descriptor 160. The first attributed failing component is the draft graph's rotary `GatherV2`, which reads the position-indexed cache inside the retained graph and raises an Ascend MTE DDR out-of-range error. C1 (`16/16`) and C4 (`64/64`) do not enter this sparse descriptor state.

Final candidate evidence (2026-08-20): query and context rotary values are populated outside capture into distinct persistent buffers, while target and draft model/attention compute remain inside genuine FULL graphs. The frozen 9B TP1 output-256 pairs completed C1/C4/C10 at 16/16 with mean accepted lengths 6.6230/6.5864/6.6957 and output-throughput ratios 1.1345/1.0099/1.0230 versus Eager. C10 exercised descriptors `160 -> 96 -> 80` without replay, `GatherV2`, MTE, contract, or request errors. Default-off boundary and layer probes localized residual branches to target layer 1 GDN numerical drift; repeated Eager controls showed equal or larger drift, so no additional runtime repair was justified.

## 3. Require complete startup capture on every local rank

- [x] 3.1 Add failing tests for a capture-manifest key containing component, local rank, graph mode, and capture descriptor.
- [x] 3.2 Make target and draft component identity explicit at wrapper construction time rather than deriving it from mutable process-global state.
- [x] 3.3 Record a manifest entry only after graph construction, warmup replay, output binding, and contract validation have all completed successfully.
- [x] 3.4 Capture target `FULL` graphs for every configured size on every TP rank before the engine becomes healthy.
- [x] 3.5 Capture draft merged-proposal `FULL` graphs for every configured size on every TP rank before the engine becomes healthy.
- [x] 3.6 Validate the local manifest after `capture_model()` and fail startup if any target/draft/rank/descriptor entry is missing or invalid.
- [x] 3.7 Add tests proving no eligible descriptor is captured lazily on the first request and arbitrary valid user-supplied capture sizes remain supported.
- [x] 3.8 Add regression tests proving manifest validation is inactive outside the exact 310P+DFlash+`FULL_DECODE_ONLY` scope.

## 4. Define genuine FULL target and draft graph boundaries

- [x] 4.1 Inventory the current target decode boundary and the draft merged-proposal boundary, explicitly listing device work that must be inside each graph and host bookkeeping that remains outside.
- [x] 4.2 Add failing tests that eligible target decode enters the target `FULL` wrapper and eligible draft proposal enters the draft `FULL` wrapper.
- [x] 4.3 Add tests that expected `NONE` states delegate to the normal eager path without capture or replay.
- [x] 4.4 Ensure the target graph contains its model/attention/device decode work and does not reduce `FULL_DECODE_ONLY` to an informational mode label.
- [x] 4.5 Ensure the draft graph contains the draft model, attention path, and proposal device compute required by the merged proposal step; graph-persistent RoPE input population remains outside replay by explicit contract.
- [x] 4.6 Add DEBUG/acceptance counters that detect graph-eligible target or draft device work executed through an eager bypass.
- [x] 4.7 Verify graph outputs and retained intermediates have stable ownership across capture and replay and are not invalidated by weak-reference or temporary-object lifetimes.

## 5. Make the graph-input contract complete and explicit

- [x] 5.1 Create a code-adjacent inventory of every retained tensor source for target and draft graphs: call arguments, forward-context tensors, attention metadata, graph parameters, and proposal-specific buffers.
- [x] 5.2 Add failing tests requiring contract providers to expose all target inputs and all draft merged-proposal inputs with a semantic role for each leaf tensor.
- [x] 5.3 Reuse the recursive contract walker only where it is mode-neutral; add explicit providers for retained state that is not reachable from call arguments.
- [x] 5.4 Declare shape, dtype, device, stride/layout, alignment, mutability, and bounded-view rules for every FULL input role; reject unknown retained tensors.
- [x] 5.5 Validate the contract immediately before capture and immediately before replay, with diagnostics naming component, rank, descriptor, semantic role, expected value, and actual value.
- [x] 5.6 Make every contract or address-safety violation fatal in INFO, DEBUG, and acceptance modes.
- [x] 5.7 Verify contract diagnostics do not call `Tensor.item()`, introduce an implicit device synchronization, or otherwise change execution behavior.

## 6. Eliminate replay-time allocation and stabilize retained addresses

- [x] 6.1 Add failing address/lifetime tests for every retained target and draft input role across repeated replays.
- [x] 6.2 Replace per-step effective-sequence-length and rejected-token temporaries with descriptor-bounded persistent buffers and in-place updates.
- [x] 6.3 Stabilize query-start locations, sequence lengths, masks, `is_prefill`, and per-layer attention metadata used by the FULL graphs.
- [x] 6.4 Stabilize token indices, D2T/proposal state, selected-token state, and other draft-specific inputs used by the merged proposal graph.
- [x] 6.5 Use bounded views into persistent storage where logical lengths vary, and prove the backing address and graph-visible metadata remain valid.
- [x] 6.6 Add a replay-path test that rejects new device tensor construction, `.to(...)`, or equivalent allocation in an eligible target or draft replay.
- [x] 6.7 Verify expected `NONE` states keep their existing eager allocation behavior and are not forced into persistent FULL buffers.
- [x] 6.8 Exercise repeated concurrency-1/concurrency-4 and prefill/decode transitions to prove buffers are neither stale nor aliased across requests.

## 7. Preserve conservative stream and lifecycle ordering

- [x] 7.1 Add lifecycle tests for capture, input update, replay, output consumption, rejection sampling, and the next draft step, including the required event/version ordering.
- [x] 7.2 Preserve the current conservative synchronization before graph reuse and any existing input-update wait until evidence proves a narrower ordering is safe.
- [x] 7.3 Ensure every target and draft input update is visible to the capture/replay stream before launch and every output consumer waits for replay completion.
- [x] 7.4 Verify all reachable cache/lifecycle transitions and explicitly classify unavailable full/partial prefix hits on the frozen Hybrid/Mamba models rather than claiming false coverage.
- [x] 7.5 Add failure diagnostics for lifecycle-version mismatches without printing per-request addresses at INFO level.
- [x] 7.6 Do not remove synchronization as a performance optimization in this change; record such work as a separately measured follow-up.

## 8. Resolve FX, W8A8, and operator compatibility from current evidence

- [x] 8.1 Re-run the smallest FDO case after dispatch/input fixes and save the first current FX/operator failure with graph, component, rank, and descriptor context.
- [x] 8.2 Reuse existing size-compatibility adaptations only when their correctness is independent of Piecewise and cover them with exact-scope tests.
- [x] 8.3 Add explicit graph-safe handling and tests for the 35B Ascend W8A8 path without changing non-FDO quantized execution.
- [x] 8.4 Reopen the Eager/Piecewise/FDO compatibility matrix for the 35B W8A8 concurrency-10 `QuantBatchMatmulV3` failure. Preserve the current non-DFlash FX size-node startup failure as separate evidence until a source-matched ordinary-FDO control is available.
- [x] 8.5 Fix a confirmed incompatibility in plugin Python only when the FULL boundary and semantics remain genuine; do not patch operators or silently route that work through eager.
- [x] 8.6 Determine whether the descriptor-160 failure has a safe plugin-Python retained-input/lifecycle correction. If it requires an upstream/operator change or eager island, stop at the failing evidence and reopen scope rather than weakening the contract.
- [x] 8.7 Add focused regressions for every independently retained graph-safe adaptation from reverted work.

## 9. Add retained observability and enforce fallback policy

- [x] 9.1 Define structured DEBUG evidence for each dispatch decision: scope predicate, execution state, chosen mode, component, rank, descriptor, capture/replay identity, and fallback reason.
- [x] 9.2 Log state transitions and expected `NONE` routing for prefill, chunked/prefix transitions, and mixed batches.
- [x] 9.3 Emit capture-manifest completion and replay counters for target and draft on every TP rank.
- [x] 9.4 Capture pre-update/post-update contract snapshots in DEBUG without changing addresses, tensor lifetimes, or stream ordering.
- [x] 9.5 Fail DEBUG and acceptance runs if an eligible uniform decode is classified as `FULL` but target or draft does not replay a valid graph.
- [x] 9.6 Keep INFO output concise and free of raw per-step addresses while retaining classified fallback summaries.
- [x] 9.7 Add unit tests proving observability is an observer only and cannot become the source of dispatch or correctness behavior.

## 10. Pass focused unit and static regression gates

- [x] 10.1 Add/extend focused 310P model-runner tests for exact-scope dispatch, state classification, startup manifest validation, and per-rank component identity.
- [x] 10.2 Add/extend DFlash proposer tests for K=15 descriptor math, persistent proposal inputs, draft graph replay, and target/draft ordering.
- [x] 10.3 Extend graph-input-contract tests for FULL-specific providers, semantic roles, alignment, retained-address validation, and fatal safety failures.
- [x] 10.4 Extend ACL graph tests for `FULL` capture/replay behavior and prove the adaptation remains inactive for unrelated modes/platforms.
- [x] 10.5 Add focused W8A8, attention-mask, rotary/position, and CausalConv regressions for every current incompatibility fixed.
- [x] 10.6 Run the complete existing 310P Piecewise test set unchanged and resolve any regression without broadening its predicate (`9 passed`).
- [x] 10.7 Run formatting, lint, type/static checks required by the repository and `git diff --check`.
- [x] 10.8 Confirm the upstream vLLM checkout is still clean and all source changes are confined to the vLLM Ascend plugin.

## 11. Extend the deterministic acceptance harness

- [x] 11.1 Refactor only mode-neutral helpers from `tools/run_310p_dflash_piecewise_acceptance.py` and add a separate FDO entry point or explicit FDO scenario configuration.
- [x] 11.2 Add preflight checks for idle cards, model availability, package/source identity, port availability, and clean process groups.
- [x] 11.3 Fix the formal dataset to GSM8K examples 0-15, output length 256, temperature 0, ignore EOS, K=15, and capture sizes `[64,32,16]`.
- [x] 11.4 Support exactly the required 9B TP1, 9B TP2, and 35B TP2 groups at concurrency 1 and 4, with gpu-memory-utilization 0.85.
- [x] 11.5 Run up to four ordinary requests as warmup, exclude them from metrics, and keep ACL internal warmup at zero.
- [x] 11.6 Capture full server logs, launch metadata, target/draft per-rank graph evidence, token outputs, speculative metrics, throughput, latency, and NPU/KV/graph memory.
- [x] 11.7 Add an evidence parser that rejects missing manifest entries, eligible eager fallback, missing replay, target/draft asymmetry, rank asymmetry, or a false-positive mode banner.
- [x] 11.8 Compare eager and FDO outputs request-by-request and compute mean accepted length, acceptance rate, request throughput, and output-token throughput with the frozen formulas.
- [x] 11.9 Guarantee cleanup of servers, benchmark clients, process groups, ports, and temporary profiling state on success or failure while preserving the evidence directory.

## 12. Run staged hardware smoke gates

- [x] 12.1 Install/build only the plugin Python package as required by the chosen editable/wheel workflow and record the imported source path; do not rebuild operators.
- [x] 12.2 Select verified idle 310P cards and run 9B TP1 startup with all production flags, DEBUG evidence, and complete `[64,32,16]` target/draft manifest validation.
- [x] 12.3 Send an ordinary 9B TP1 request and prove prefill/transition runs `NONE` while uniform decode replays `FULL` for both target and draft on rank 0.
- [x] 12.4 Exercise prefix miss, chunked-prefill, concurrency, and prefill/decode transitions; preserve the frozen Hybrid/Mamba model's measured 0% hit rate as capability evidence for unavailable full/partial hits.
- [x] 12.5 Run 9B TP2 startup/request smoke and prove identical target/draft manifest and replay coverage on both ranks.
- [x] 12.6 Run 35B TP2 W8A8 startup/request smoke and prove identical target/draft manifest and replay coverage on both ranks.
- [x] 12.7 At every failed smoke gate, preserve the first causal traceback and return to the responsible task instead of continuing to later acceptance runs.

## 13. Execute the six formal eager/FDO acceptance pairs

- [x] 13.1 Run and archive 9B TP1 concurrency-1 eager and FDO results under identical frozen inputs and generation settings.
- [x] 13.2 Run and archive 9B TP1 concurrency-4 eager and FDO results under identical frozen inputs and generation settings.
- [x] 13.3 Run and archive 9B TP2 concurrency-1 eager and FDO results under identical frozen inputs and generation settings.
- [x] 13.4 Run and archive 9B TP2 concurrency-4 eager and FDO results under identical frozen inputs and generation settings.
- [x] 13.5 Run and archive 35B TP2 concurrency-1 eager and FDO results under identical frozen inputs and generation settings.
- [x] 13.6 Run and archive 35B TP2 concurrency-4 eager and FDO results under identical frozen inputs and generation settings.
- [x] 13.7 Run Piecewise as a non-blocking performance reference without substituting it for any eager/FDO acceptance pair; retain its independent 35B C10 failure as invalid partial evidence.

## 14. Evaluate correctness, graph truth, performance, and memory gates

- [x] 14.1 Require 16/16 successful requests and apply the approved deterministic-output policy: exact tokens by default; repeated controls and layer evidence may classify a single isolated floating-point wording branch as non-blocking, while a stable multi-request branch cluster remains blocking.
- [x] 14.2 Require mean accepted length at least 5.0 and at least 90% of eager for every formal group.
- [x] 14.3 Require FDO acceptance rate to be no more than 5 percentage points below eager for every formal group.
- [x] 14.4 Require FDO request throughput and output-token throughput to each be at least 85% of eager for every formal group.
- [x] 14.5 Require evidence that prefill/transition/mixed work used `NONE`, eligible uniform decode used `FULL`, and target plus draft replayed on every rank and configured descriptor.
- [x] 14.6 Record startup graph memory, peak NPU memory, and KV-cache capacity for eager and FDO at gpu-memory-utilization 0.85 and investigate unexplained regressions.
- [x] 14.7 Produce a machine-readable and human-readable acceptance report linking every result to its launch command, logs, outputs, and graph evidence.

Final probes-disabled validation evidence (2026-08-20) is archived under
`/home/whn/aisbench_runs/fdo_context_extent_acceptance_20260820`,
`/home/whn/aisbench_runs/fdo_context_extent_acceptance_repeat_20260820`,
`/home/whn/aisbench_runs/fdo_final_acceptance_apply2_20260820`, and
`/home/whn/aisbench_runs/fdo_final_acceptance_repeat_20260820`. Every row used
GSM8K records 0-15, output length 256, temperature 0, ignore EOS, DFlash K=15,
and a fresh Eager/FDO service pair. The C10 extension used capture sizes
`[160,64,32,16]`; the six formal groups used `[64,32,16]`.

| Group | FDO success | FDO output tok/s | vs eager | FDO accepted length | vs eager | differing requests |
|---|---:|---:|---:|---:|---:|---:|
| 9B TP1 C1 | 16/16 | 31.4817 | 113.45% | 6.6230 | 100.00% | 0/16 |
| 9B TP1 C4 | 16/16 | 83.1936 | 100.99% | 6.5864 | pass | repeat drift; layer-localized |
| 9B TP1 C10 | 16/16 | 99.8461 | 102.30% | 6.6957 | pass | repeat drift; layer-localized |
| 9B TP2 C1 | 16/16 | 46.0096 | 160.51% | 6.6115 | pass | 0/16 |
| 9B TP2 C4 | 16/16 | 113.1781 | 117.03% | 6.5204 | 98.62% | 0/16 |
| 35B TP2 C1 | 16/16 | 41.9232 | 219.09% | 6.9699 | 99.52% | one isolated branch after repeat |
| 35B TP2 C4 | 16/16 | 82.9410 | 162.96% | 6.7198 | 96.62% | one stable isolated branch after repeats |

All seven FDO rows contain real target and draft native graph files, complete
per-rank manifests, runtime FULL replays for every configured descriptor,
expected NONE dispatches for non-eligible work, and zero operational traceback,
contract, GatherV2, MTE, or ACL replay errors. C10 exercised descriptor 160 and
runtime contraction to 80 tokens without the historical address fault. FDO
graph memory was approximately 0.91 GiB for 9B TP1, 0.92-0.97 GiB per rank for
9B TP2, and 0.93 GiB per rank for 35B TP2; KV-cache capacity remained within
0.02 GiB of eager in the recorded startup summaries.

Repeated Eager and FDO controls plus layer probes classified the apparent
multi-request mismatch sets as run-to-run numerical drift. For 9B TP1 C4, the
first target difference occurs inside layer 1 GDN with Eager-to-FDO max-absolute
differences of about `1.9e-5` to `3.1e-5`, while independent Eager runs differ by
about `3.1e-5` to `3.8e-5`. In 35B TP2 C4, only request 6 at token 140 remained
stable across both FDO runs and both Eager controls; C1 likewise retained only
one isolated branch after repeat classification. These are recorded under the
user-approved isolated wording-branch policy and do not authorize another FDO
runtime change.

## 15. Profile the accepted implementation and prepare handoff

- [x] 15.1 Defer the 9B TP1 three-mode profiler comparison to a separate performance change; it is not a correctness or acceptance gate for this repair.
- [x] 15.2 Defer the 35B TP2 three-mode profiler comparison to the same separate performance change; the frozen throughput gates are complete.
- [x] 15.3 Record profiling dimensions (target, draft, attention, rejection, replay, synchronization, host scheduling, communication) as follow-up scope without changing this accepted implementation.
- [x] 15.4 Separate correctness blockers from follow-up performance opportunities; do not remove conservative synchronization or expand scope during this change.
- [x] 15.5 Re-run focused tests (`271 passed`), the unchanged Piecewise regression set (`9 passed`), formatting/static checks, `git diff --check`, and upstream-vLLM cleanliness checks from a fresh shell.
- [x] 15.6 Review the final plugin diff for exact-scope guards, absence of operator/upstream edits, absence of eager islands, and absence of unrelated refactors.
- [x] 15.7 Clean all test services and temporary profiler state, preserve acceptance artifacts, and present the implementation diff plus evidence for user review before any commit or push.

## 16. Close 35B W8A8 concurrency-10 repeated replay

- [x] 16.1 Freeze the TP2/TP4 10-of-20 failure commands, first-cause logs, graph manifests, and partial metrics under one dated repair evidence root; mark partial throughput and acceptance metrics invalid.
- [x] 16.2 Reproduce TP2 concurrency 10 twice unchanged and run paired Eager and released-Piecewise controls; keep the current non-DFlash FX startup failure separate.
- [x] 16.3 Add bounded default-off diagnostics for target/draft, rank, descriptor, replay generation, model layer/operator, active rows, activation/weight/scale/workspace identities, MoE routing summary, and stream/lifecycle versions.
- [x] 16.4 Compare descriptor 16, first descriptor-160 replay, and failing descriptor-160 reuse; record one falsifiable root-cause hypothesis before editing production code.
- [x] 16.5 Add and observe a focused failing regression that models the proven retained-input, alias, routing, workspace, or lifecycle defect without mocking away the failure.
- [x] 16.6 Implement the smallest exact-scope plugin-Python fix; do not edit upstream vLLM, operators, C++/AscendC, Eager, or released Piecewise behavior.
- [x] 16.7 Pass 35B TP2 concurrency 10 with 20/20 requests, output 256, DFlash K=15, capture sizes `[160,16]`, genuine target/draft replay, and no QuantBatchMatmul/L0C/ACL/HCCL failure.
- [x] 16.8 Pass the identical 35B TP4 concurrency-10 gate on every rank.
- [x] 16.9 Re-run paired Eager and Piecewise controls and reject any success, acceptance-length, output-throughput, dispatch, or graph-authenticity regression.
- [x] 16.10 Run the complete frozen mode-first matrix plus one AISBench smoke per successful deployment, archive throughput and accepted-length deltas against Eager, and retain failed Piecewise deployments without treating partial metrics as valid.
- [x] 16.11 Complete the still-open repeated-replay allocation/aliasing and lifecycle ordering tasks 6.6 and 7.1 with the new regression evidence.
- [x] 16.12 Run focused/full unit tests, static checks, `git diff --check`, OpenSpec strict validation, upstream cleanliness, process/card cleanup, and final code review before marking this change complete.

Piecewise A/B evidence (2026-08-20): the exact user flags, `[160,64]`
capture sizes, and frozen C10/20 workload completed only 10/20 both with the
accepted FDO production delta and after reversing only that delta. Both runs
entered real Piecewise capture/replay before ACL 507015. The candidate is
therefore not a Piecewise behavior regression; the independently reproducible
Piecewise failure remains outside this FDO closure and its partial metrics are
not used as a performance baseline.

Repeated-replay unit evidence: `test_dflash_full_decode_acl_graph.py` now
rejects device tensor constructors during an eligible replay and asserts the
contract-validation, current-stream synchronization, graph launch, and output
consumption order. The focused file passes 11/11 tests.

Final mode-first closure (2026-08-21) is archived under
`/home/whn/vllm_repair/final_matrix_20260821/`. Current FDO passed 4B TP1,
4B TP2, 35B W8A8 TP2, and 35B W8A8 TP4 at C1 (4/4) and C10 (20/20), with one
AISBench smoke per deployment. Output throughput was 98.09%-306.72% of the
frozen Eager rows and accepted length was 97.68%-100.00% of Eager. Every rank
has complete target/draft manifests for descriptors 16 and 160 and real replay
evidence. The full software gate passed 362 tests with 2 skips; Ruff,
`git diff --check`, OpenSpec strict validation, upstream cleanliness, and
service/port cleanup passed. Prefix miss/chunked/mixed/decode states were
observed; the frozen Hybrid/Mamba models reported a 0% prefix-cache hit rate,
so full/partial hit is recorded as an unavailable experimental capability.
Detailed ratios and artifact links are in `summary.md` in that evidence root.

## 17. Final integration and scope isolation

- [x] 17.1 Freeze the final reproducible matrix as 4B TP1/TP2 and 35B W8A8 TP2/TP4, each at C1 (4 requests) and C10 (20 requests), output 256, GSM8K original order, and capture sizes `[160,16]`.
- [x] 17.2 Update the repository acceptance runner and unit tests to reject scenarios outside that matrix and to use the correct request count for each concurrency.
- [x] 17.3 Scope descriptor padding, attention-mask padding, slot arithmetic, and sequence-length arithmetic to exactly `310P + DFlash + FULL_DECODE_ONLY`; preserve the legacy Eager/Piecewise and non-DFlash behavior with focused regression tests.
- [x] 17.4 Exclude the abandoned Piecewise W8A8 Candidate B/custom-operator work and all local evidence, backup, agent, and configuration files from the integration diff.
- [x] 17.5 Complete an independent final code review; resolve all Important scope and reproducibility findings before commit.

Post-review verification (2026-08-22): the related 310P suite passed 369 tests
with 2 skips after exact-scope isolation and harness synchronization. Fresh
server-1 FDO gates using the formatted integration diff passed 4B TP1 C1
(4/4, 51.68 output tok/s, accepted length 7.32) and C10 (20/20, 165.49
output tok/s, accepted length 6.59), plus 35B W8A8 TP2 C1 (4/4, 43.39
output tok/s, accepted length 6.47) and C10 (20/20, 120.66 output tok/s,
accepted length 6.58). Both deployments recorded real ACL graph replay and
completed without graph-contract, ACL 507xxx, AICore, or HCCL failures.
