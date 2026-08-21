## Context

See `proposal.md` for motivation and `specs/310p-dflash-full-decode-only-graph/spec.md` for the behavioral contract.

The implementation baseline is vLLM Ascend `959b9a6a`, which contains the verified 310P DFlash Piecewise repair and its persistent-buffer, graph-input-contract, guarded FX, W8A8, and acceptance foundations. Those foundations are reusable evidence, not permission to broaden the Piecewise activation predicate. Reverted FULL-family commits remain diagnostic references only.

In frozen vLLM 0.24, `FULL_DECODE_ONLY` is a composite mode whose decode mode is `FULL` and mixed mode is `NONE`. The dispatcher selects FULL only for uniform decode. For DFlash K=15, the uniform query length is 16 tokens per request; capture sizes 16, 32, and 64 therefore represent padded request capacities 1, 2, and 4. Startup capture is driven by the upstream dispatcher descriptors, while runtime execution reaches plugin-owned target and draft `ACLGraphWrapper` instances with concrete `BatchDescriptor` values.

The current plugin already contains partial FULL infrastructure: target FULL wrapping, a draft wrapper around the merged proposal callable, dummy FULL attention metadata, graph-parameter workspaces, and conservative current/update stream synchronization. It does not yet provide one exact 310P+DFlash+FULL_DECODE_ONLY policy, a complete per-component/per-rank startup manifest, or a FULL-specific contract covering tensors retained through forward-context and proposer-owned buffers. Some comments and branches still assume the drafter is Piecewise-only. These ambiguities must be resolved from observed execution rather than patched around with mode coercion.

Historical 9B evidence is intentionally inconclusive: one earlier branch captured and replayed FULL decode, while another failed inside capture with an unavailable `aclnnCausalConv1dV310` symbol. The first implementation task is therefore a fresh reproduction on the frozen current stack. A historical success does not waive current safety checks, and a historical missing-symbol failure does not authorize operator changes.

## Goals / Non-Goals

**Goals:**

- Implement one independently gated 310P+DFlash+FULL_DECODE_ONLY policy over the current baseline.
- Make FULL/NONE routing a closed, testable state machine driven by the actual DFlash batch, not by log messages or requested mode alone.
- Capture and replay the complete target and merged DFlash draft computation for every configured uniform descriptor on every TP rank.
- Extend stable ownership, address, bounds, alignment, and stream-order contracts to all tensors retained by a FULL graph, including inputs reached through forward context or proposer state.
- Preserve conservative synchronization until correctness tests and profiles prove an optimization safe.
- Produce structured evidence that distinguishes expected NONE work from unexpected eager fallback and proves genuine FULL replay.
- Reuse the existing acceptance harness and verified generic utilities without changing Piecewise behavior.

**Non-Goals:**

- Enabling or repairing `FULL`, `FULL_AND_PIECEWISE`, or any non-DFlash FULL path.
- Modifying frozen upstream vLLM, torch-npu, CANN, C++, AscendC, or a custom operator.
- Creating eager islands inside target or draft FULL uniform decode.
- Requiring FULL_DECODE_ONLY to outperform Piecewise in the first functional change.
- Removing synchronization as a speculative optimization.
- Completing unrelated unchecked tasks from the Piecewise OpenSpec.

## Decisions

### 1. Add an independent exact-scope policy

A plugin-owned policy will activate only when all of these conditions hold:

- current platform is Ascend 310P;
- speculative method is `dflash`;
- requested and resolved mode is exactly `FULL_DECODE_ONLY`;
- the active configuration is the frozen supported vLLM interface when a private compatibility hook is used.

The policy will consume the active `VllmConfig`; it will not use a mutable process-global switch and will not add a new environment variable. Mode-neutral utilities may be extracted from the Piecewise implementation, but `is_310p_dflash_piecewise()` remains Piecewise-only and the new policy remains FULL_DECODE_ONLY-only.

Alternative considered: change the Piecewise predicate to accept every graph mode containing Piecewise or FULL. Rejected because it couples activation, makes mode regressions difficult to isolate, and could accidentally enable future `FULL_AND_PIECEWISE` work.

### 2. Represent runtime routing as a closed state machine

The 310P runner will classify each batch before dispatcher selection using observable scheduler and attention state:

- `EXPECTED_NONE_PREFILL`: initial or ordinary prefill;
- `EXPECTED_NONE_CHUNKED_PREFILL`: chunked prefill;
- `EXPECTED_NONE_PREFIX_TRANSITION`: prefix-cache hit/miss transition;
- `EXPECTED_NONE_MIXED`: mixed prefill/decode;
- `FULL_ELIGIBLE_UNIFORM_DECODE`: every active request is decode, every request has query length 16, token count equals request count times 16, and no other dispatcher constraint disables FULL;
- `UNSUPPORTED_UNIFORM_DESCRIPTOR`: uniform decode is structurally eligible but cannot map safely to a configured descriptor;
- `MODE_MISMATCH` or `SAFETY_FAILURE`: the selected runtime mode or retained inputs violate the policy.

Expected NONE states pass `force_eager=True` or accept dispatcher `NONE`. FULL-eligible state is offered to the dispatcher without coercing the result. The returned runtime mode and descriptor are then verified. This preserves upstream ownership of padding and descriptor creation while making an unexpected selection observable.

Alternative considered: force `CUDAGraphMode.FULL` whenever attention state is speculative decode. Rejected because it bypasses dispatcher constraints, can replay an incompatible descriptor, and would make mixed or cache-transition batches unsafe.

### 3. Use exact K=15 descriptor arithmetic without hard-coding user configuration

The formal capture list is `[64,32,16]`. With query length 16, these descriptors correspond to padded request capacities 4, 2, and 1. The policy validates divisibility by the active uniform query length, request capacity, maximum sequence settings, and dispatcher-produced `BatchDescriptor` fields.

Implementation will read the resolved capture list and will not insert, remove, or reorder sizes. Unsupported user sizes remain the dispatcher's concern unless they are selected for graph-eligible work, at which point validation reports the exact mapping failure. Tests cover `[64,32,16]`, alternative valid configured lists, missing 16, and non-divisible values without inventing a replacement descriptor.

### 4. Build a local startup capture manifest and fail initialization when incomplete

Each worker rank will maintain a startup manifest keyed by:

`(scope, component, rank, runtime_mode, BatchDescriptor)`.

Target and draft wrappers register capture completion only after the ACL graph object and retained output are valid. After upstream `capture_model()` returns, the 310P runner validates that every configured FULL decode descriptor has one target and one draft capture on the local rank. Engine initialization already waits for all workers; a missing local manifest entry raises on that worker and prevents healthy engine startup. No new cross-rank collective is placed in the hot path.

Formal validation aggregates per-rank manifests from structured worker logs or snapshots. Runtime replay increments the matching manifest entry. Startup capture counts alone never satisfy acceptance.

Alternative considered: verify capture only in the external harness after the API becomes healthy. Rejected because it permits a misleading healthy service and weakens the requirement that graph startup itself is genuine.

### 5. Capture the merged draft proposal, not only the draft model module

The draft FULL boundary remains the merged DFlash proposal callable. It includes context KV preparation, query construction that must be device-graph-safe, draft model forward/attention, logits/sampling, and draft-to-target vocabulary remapping where applicable. Wrapping only the draft model module would leave material proposal work eager while presenting a draft FULL counter.

The target FULL wrapper and draft merged wrapper both use runtime mode `FULL`. During expected NONE work they delegate directly to their runnable. During FULL work they capture or replay the exact descriptor. Component identity is attached at wrapper creation or explicit call context rather than inferred from a mutable global execution policy.

Host scheduling metadata creation and persistent-buffer copies stay outside the graph. They are not model eager islands, but their completion is part of the replay precondition.

### 6. Extend the graph-input contract through explicit providers

The existing recursive contract covers positional and keyword tensor arguments. FULL graphs also retain tensors reachable through forward context, attention metadata, graph-parameter workspaces, closure state, and proposer-owned persistent buffers. Generic traversal of every object in the process would be unstable and expensive, so each FULL component will expose an explicit contract provider.

The target provider inventories model arguments plus graph-retained attention and runner state. The draft provider inventories merged-proposal arguments plus DFlash input IDs, positions, hidden/context/query buffers, block tables, slot mappings, sequence/query metadata, rejection counts, mask/workspace tensors, sampling indices, and graph-parameter workspaces. Every role declares ownership and the alignment source from its consumer.

Capture stores pointer, base storage, offset, accessible bytes, dtype, shape, stride, contiguity, device, natural alignment, declared operation alignment, component, rank, and descriptor. Replay captures the same roles and compares them before invoking ACL replay. Unknown required alignment or missing roles fail validation instead of being guessed.

Alternative considered: rely on the existing flat positional-address assertion. Rejected because most FULL-retained attention and draft tensors are not top-level positional arguments.

### 7. Reuse descriptor-bounded persistent storage and eliminate replay-time allocation

Graph-eligible inputs will be allocated during runner/proposer initialization or in a bounded descriptor cache and updated in place. The audit covers all roles listed by the contract providers. Particular attention is required for expressions that currently create new tensors or views during DFlash metadata preparation, including effective sequence lengths, rejected-token handling, per-layer slot mappings, query boundaries, boolean prefill metadata, and dtype/device conversions.

CPU-only objects that are consumed solely before replay may remain dynamic. A CPU object whose device conversion or value is retained by the graph must write into a persistent device buffer. A variable-length view is allowed only when its base storage, offset, maximum bounds, and descriptor-specific shape are captured and stable.

The Piecewise persistent rejection-count and related buffers may be reused when their ownership contract is mode-neutral. Any Piecewise-only enable hook will be refactored into a mode-neutral implementation plus separate callers rather than expanded implicitly.

### 8. Preserve conservative stream and lifecycle ordering

The first repair keeps the existing correctness barriers and makes their required order explicit:

1. drain prior replay before CPU block-table condense or row movement;
2. update host snapshots for the current batch;
3. copy persistent target and draft inputs to device/update stream;
4. update FULL attention graph parameters and workspaces;
5. make the replay stream wait for the update stream;
6. validate the retained-input contract;
7. replay target or draft graph;
8. expose output only after the existing consumer ordering permits it.

Tests use mocked events and observable state versions to prove replay cannot see the prior iteration. Hardware lifecycle tests cover `1 -> 4 -> 1`, partial completion, condense, prefix-cache hit/miss, and repeated descriptors. Only the post-acceptance profile may motivate a separate synchronization optimization; the implementation plan will not remove a barrier merely because it appears expensive.

### 9. Keep FX and operator compatibility evidence-driven

The exact fresh FULL_DECODE_ONLY startup is rerun before compatibility changes. If scalar FX `size(dim)` decomposition fails, the verified transformer may be refactored into a shared implementation with independent Piecewise and FULL_DECODE_ONLY guards. Private helper version/signature checks and graph-form rejection remain mandatory. No helper is installed for this scope unless the exact path reproduces the need.

If `aclnnCausalConv1dV310` fails, validation first distinguishes:

- symbol/library unavailable in eager too: frozen environment defect, not a graph repair;
- eager and Piecewise work but FULL capture fails: investigate graph input shape, workspace, dispatch branch, capture context, and supported operation contract in plugin Python;
- operation captures but replay fails: investigate retained storage and update ordering.

No result permits replacing the operator, editing its implementation, or excluding it from the FULL graph. If a safe plugin-side FULL contract is impossible, the change is reported blocked for a separate scope decision.

### 10. Generalize graph-safe Python adaptations only through explicit callers

The 35B W8A8 path and any Python-level layout or dynamic-shape adaptations proven by Piecewise may be reused only after tests demonstrate they are graph-mode-neutral. Activation functions will accept an explicit policy or have separate exact-scope entry points. Imports alone must not mutate unrelated execution.

The same rule applies to address diagnostics and size-node compatibility: share the algorithm, not the activation condition. Out-of-scope unit tests snapshot delegation and resolved configuration before and after initialization.

### 11. Store evidence with wrapper entries, not execution-control counters

Every FULL graph entry owns capture/replay counters and a diagnostic record. A weak wrapper registry may aggregate snapshots, but counters and snapshots never choose execution behavior. Required records include:

- scope and component;
- rank and descriptor;
- requested, resolved, wrapper, and runtime modes;
- configured capture sizes;
- capture/replay counts and startup-manifest status;
- previous/current mode and transition reason;
- fallback reason and count;
- contract comparison and alignment outcome;
- update/replay ordering milestones.

DEBUG emits capture, first replay, transitions, contract failures, and requested snapshots. INFO keeps concise startup and error records plus lightweight metrics, without printing addresses per iteration. The current Piecewise-specific debug selector is not extended into a new undocumented environment variable.

### 12. Treat fallback and safety as separate policies

Expected NONE routing is normal FULL_DECODE_ONLY behavior. Unsupported or mismatched uniform decode is an unexpected fallback. Under DEBUG/acceptance it raises immediately so a completed request cannot masquerade as graph success. INFO may execute eager for availability while incrementing a machine-parseable reason counter.

Safety failures are different: changed addresses, out-of-bounds views, incompatible layout, unknown alignment, stale metadata version, or broken stream ordering always reject replay in every log level. They are not converted to eager because doing so could hide memory corruption or correctness failures.

### 13. Extend the acceptance harness instead of creating an unrelated runner

The existing 310P DFlash Piecewise harness will be refactored only where its mechanics are mode-neutral: frozen manifest checks, idle-card selection, process-group ownership, health waiting, warmup exclusion, GSM8K ordering, token comparison, metric collection, and cleanup. FULL_DECODE_ONLY adds a mode-specific evidence parser and capture manifest validator.

Each paired run uses DFlash eager as the correctness and quantitative baseline. Piecewise is a secondary reference and does not control pass/fail. The harness checks the six agreed groups, configured features, `[64,32,16]`, startup memory, target/draft per-rank capture/replay, expected NONE transitions, zero unexpected validation fallbacks, token equality, acceptance thresholds, and throughput threshold.

The harness kills only its recorded process group and proves port/card release. It does not stop user-owned services or reuse stale metrics from another run.

### 14. Separate formal acceptance from post-acceptance profiling

Formal metrics use up to four unmeasured ordinary warmup requests and then GSM8K records 0-15 with output 256, temperature 0, ignore EOS, and fixed order/seed. ACL graph internal warmups stay at zero. Counter snapshots before warmup, after warmup, and after formal traffic make graph activity attributable while all performance and acceptance calculations use only formal requests.

After all functional gates pass, 9B TP1 C1 and 35B TP2 C4 are profiled in eager, Piecewise, and FULL_DECODE_ONLY modes. The comparison attributes time to host preparation, synchronization, target replay, draft proposal/replay, and sampling. It informs later optimization but does not retroactively weaken correctness or require a positive Piecewise delta.

### 15. Use staged test gates and stop on the first unexplained failure

Implementation advances through these gates:

1. scope, state-machine, descriptor, and out-of-scope unit tests;
2. startup manifest, wrapper evidence, fallback, and contract-provider tests;
3. persistent-buffer and stream/lifecycle tests for target and draft;
4. focused existing 310P runner, attention, DFlash, W8A8, compilation, and Piecewise regressions;
5. 9B TP1 startup capture plus NONE-to-FULL replay smoke;
6. 9B TP2 and 35B TP2 startup smoke;
7. all six eager/FULL_DECODE_ONLY formal pairs with Piecewise references;
8. two non-blocking three-mode profiles;
9. final source, package, artifact, process, port, and card audit.

Each failure returns to the smallest reproducible gate. Tests, capture sizes, serving flags, or thresholds are not weakened to obtain a pass.

### 16. Roll back the concurrency-10 delta and debug from the first divergence

The accepted FDO implementation before concurrency-10 work is the control state. The current state, which completes concurrency 10 but lowers 9B acceptance length from about 7.48 to about 3.28, is a known-regressed experiment rather than a new baseline. Its diff and logs remain evidence, but its position-tail and draft-RoPE adaptations are removed before root-cause work resumes.

Rollback is surgical: reverse only changes introduced after the original concurrency-10 failure was observed, while preserving the FDO scope policy, target/draft FULL wrappers, graph contracts, observability, and every unrelated user change. The recovered state must demonstrate both properties before a new fix is attempted:

1. concurrency 1 and 4 recover their deterministic output, acceptance length, and throughput under the frozen 9B TP1 workload;
2. concurrency 10 reproduces the original failure rather than the later quality regression.

The investigation follows one active-lane data path:

`scheduler positions -> persistent position buffer -> draft RoPE inputs -> draft logits/tokens -> rejection sampler`

Default-off DEBUG probes compare the recovered control and one candidate at the first differing iteration. Padding is allowed to initialize inactive descriptor tails, but it must be observationally invisible to every active lane. Moving a gather or other position-dependent computation outside the graph is not accepted merely because capture succeeds; its update ordering and active-lane values must match the unpadded reference.

Hypotheses are tested one at a time. The current external draft-RoPE precompute is the first high-risk boundary to inspect because it changed active position-dependent data for all concurrency levels, but it is not assumed to be the root cause until the first-divergence evidence proves it. Failed synchronization, staging, pointer, and padding experiments remain diagnostic references only.

Hardware validation is tiered to keep iteration fast:

1. focused unit tests for active-lane identity, tail initialization, and update ordering;
2. a four-prompt 9B TP1 gate at concurrency 1 and 10 with output length 256;
3. only after that passes, the 16-prompt 9B TP1 concurrency-1/concurrency-4/concurrency-10 gate with exact output, acceptance, throughput, graph-replay, and error checks;
4. only after all 9B gates pass, resume TP2 and 35B work.

Alternatives considered: retain the current concurrency-10 patch and tune acceptance afterward. Rejected because a 50%+ acceptance-length loss is a correctness-quality regression and obscures the original failure. Re-run the entire model matrix after each edit. Rejected because it slows causal iteration without improving the 9B first-failure signal.

### 17. Reopen 35B W8A8 descriptor-160 replay as a separate causal closure

The latest fixed matrix supersedes the earlier claim that no operator failure remained. On both TP2 and TP4, 35B W8A8 FDO captures real target and draft graphs and passes descriptor 16 at concurrency 1, but concurrency 10 completes only 10 of 20 requests before `QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21` raises an AICore exception. TP4 additionally reports an L0C read/write conflict, followed by ACL 507015 and HCCL ERR02005. Eager completes the paired workload, and the user reports the released Piecewise path completes it, so the failure is treated as an FDO repeated-replay defect until evidence proves otherwise.

The investigation is intentionally narrow:

1. reproduce TP2 twice with the frozen 20-request concurrency-10 command and preserve the first causal device/runtime line;
2. run the identical Eager and released-Piecewise controls and retain ordinary non-DFlash FDO only as a diagnostic reference;
3. add bounded, default-off diagnostics identifying component, rank, model layer/operator, descriptor, replay generation, active rows, activation/weight/scale/workspace identities, MoE routing summary, and stream/lifecycle versions;
4. compare descriptor 16, the first successful descriptor-160 replay, and the first failing descriptor-160 reuse;
5. state one falsifiable root-cause hypothesis, then write and observe a focused failing regression before changing production code;
6. implement the smallest plugin-Python, exact-scope correction; no operator edit, upstream edit, eager island, or broad synchronization is allowed;
7. gate hardware in the order TP2 C10, TP4 C10, paired Eager/Piecewise controls, then the complete frozen matrix and AISBench smoke.

The ordinary non-DFlash control is not conflated with this failure. Historical 310P Qwen3.5 FDO includes a verified synchronize-on-finished-request guard for block-table condense, and that guard remains in the current source. A current staged-source non-DFlash control instead failed before readiness in upstream FX size-node decomposition (`Node size ... still had users`). Until a clean historical/source-matched control is run, that is a separate compatibility observation, not evidence for or against the DFlash W8A8 replay hypothesis.

The first working hypothesis may concern stale or aliased graph-retained MoE/quantization state only after evidence shows an identity, value, routing, workspace, or lifecycle mismatch on the failing reuse. The mere facts that descriptor 160 is involved and the failure occurs after 10 requests are not sufficient to authorize a buffer, padding, or synchronization change.

Two exact-scope eager-fallback candidates were rejected by the unchanged TP2 hardware gate. Keeping FDO W8A8 linear matmuls in two-way chunks outside profile/capture passed its RED/GREEN branch test but reproduced the same L0C/507015 failure. Restoring normal MoE events whenever FDO dispatch returned to `NONE` also passed its RED/GREEN lifecycle test but reproduced the same failure. Both candidates were fully reverted; their logs remain under the dated repair evidence root.

The next falsifiable hypothesis moves the causal boundary back to the preceding dual-graph replay. The existing finished-request guard synchronizes only the current NPU stream before block-table condense, while exact DFlash FDO has independent target and draft ACL graphs with internal graph streams. The hypothesis is that replay returns before all child graph work is joined, so the first replenishment `ChunkedPrefill` launches an eager W8A8 quant matmul while a descriptor-160 graph quant matmul is still live, producing the rank-varying L0C scheduling conflict. This explains why changing the new eager batch's chunking and MoE events had no effect and why the error surfaces at a later copy/event synchronization rather than as a wrapper contract error.

The control is staged before another production edit: ten prompts at concurrency ten must complete without queue replenishment, while twenty prompts at concurrency ten must reproduce at the first replacement prefill. A device-wide completion barrier may be used once as a diagnostic oracle on that transition, but it is not an acceptable final fix. If the oracle passes, the implementation must establish a targeted target/draft graph-completion dependency and retain the existing current-stream behavior for Eager, Piecewise, and ordinary FDO.

A candidate is rejected immediately if it:

- changes active-lane logits/tokens or materially lowers acceptance length;
- changes Eager or Piecewise behavior;
- avoids the failing operator by routing work eagerly;
- passes capture but lacks real target/draft replay on every rank;
- needs an operator, C++, or upstream vLLM modification.

The stream-completion oracle was falsified: a device-wide barrier at the
replacement transition reproduced the same 10-of-20 failure. A focused FX
inspection then exposed the actual execution mismatch. FDO profiles one
dynamic compiled target callable while the W8A8 graph-safe flag is active;
that profile freezes the two-way symbolic quant-matmul split into the FX
graph. `ACLGraphWrapper` correctly delegates a replenishment batch with
runtime mode `NONE`, but the delegated runnable is still that same
FULL-profile compiled callable. At the first replacement prefill it therefore
executes the FULL-profile W8A8 topology at 242 active rows. Eager uses its
ordinary size-sensitive path, and Piecewise owns a different graph boundary.

This hypothesis was tested before the production change. A RED policy test
required only runtime `NONE` in the exact 310P + DFlash + FDO scope to bypass
the FULL-profile compiled callable, while profile runs, runtime `FULL`, Eager,
Piecewise, other methods, and other platforms remained unchanged. A separate
row-alignment candidate changed the compiled FX chunks from 121+121 to
128+128 and still reproduced the same QBM/L0C failure, so alignment alone was
rejected and fully reverted.

The accepted correction routes the complete expected-`NONE` target and draft
batch through the existing uncompiled model entry. This is the native
prefill/mixed execution half of `FULL_DECODE_ONLY`, not an eager island inside
eligible FULL decode: every uniform decode iteration still dispatches FULL and
replays both target and draft graphs. DEBUG evidence records
`execution=expected-none-uncompiled graph_eligible=false` for this boundary.
Two unchanged TP2 C10 runs completed 20/20 at 122.62 and 120.90 output tok/s
with accepted lengths 6.57 and 6.54. TP4 C10 completed 20/20 at 131.82 tok/s
with accepted length 6.54. Both descriptors 160 and 16 showed genuine target
and draft replay on every participating rank, with no QuantBatchMatmul, L0C,
507015, HCCL, or graph-input-contract failure. Evidence is archived under
`/home/whn/aisbench_runs/fdo_w8a8_c10_repair_20260820/`.

The Piecewise guard was closed with an exact source A/B rather than inferred
from the activation predicate. With the accepted two-file production delta
present, the frozen 35B TP2 Piecewise command captured and replayed genuine
graphs but completed only 10/20 requests before ACL 507015. The production
delta was then saved, reversed without changing tests or launch flags, and the
identical Piecewise deployment and workload were rerun. It again completed
10/20 with the same replacement-batch failure class. The accepted production
delta was immediately restored and its focused tests rerun. This proves the
Piecewise C10 failure is not introduced by the FDO expected-NONE routing fix;
it remains an independently reproducible 35B W8A8 + DFlash + Piecewise mixed-
batch issue and its partial throughput/acceptance values are invalid. Evidence
is under `tp2_control_piecewise_exact_user_flags/` and
`tp2_control_piecewise_candidate_reversed/` inside the dated repair root.
## Risks / Trade-offs

- **FULL retains more state than Piecewise, so an incomplete contract can replay stale or reallocated inputs** -> Use explicit target/draft contract providers, persistent storage, pre-replay validation, and lifecycle transitions before formal traffic.
- **A concurrency-10 crash can be hidden by tail initialization that also corrupts active draft positions** -> Compare active-lane position/RoPE/draft outputs against the recovered control and fail the fast gate before broader validation.
- **The concurrency-10 delta is interleaved with uncommitted FDO work** -> Preserve a patch of the regressed state, reverse only evidence-backed delta hunks, and verify the base FDO files and unrelated user changes remain intact.
- **Conservative synchronization may limit the expected FULL performance gain** -> Keep it for correctness, measure two representative profiles, and make synchronization optimization a later evidence-backed change.
- **Startup capture for target and draft across three descriptors increases graph memory** -> Record per-run graph memory and KV capacity, share the global pool where supported, remove duplicate ownership, and fail rather than drop required descriptors.
- **Historical `aclnnCausalConv1dV310` behavior may differ across installed stacks** -> Reproduce on the frozen current stack, distinguish missing library from capture-only incompatibility, and prohibit operator changes or eager islands.
- **Refactoring generic Piecewise utilities could regress the working mode** -> Separate algorithms from activation predicates, add explicit out-of-scope delegation tests, and run the Piecewise regression/acceptance smoke before FULL formal validation.
- **The draft merged callable contains more than the draft model and may expose new dynamic allocations** -> Inventory every retained role, eliminate allocations only in FULL-eligible paths, and keep expected NONE paths unchanged.
- **INFO eager fallback favors availability and can hide low graph coverage in casual testing** -> Require DEBUG/acceptance fail-closed behavior and per-rank manifest/replay evidence for every formal result.
- **TP logs can be incomplete if only rank 0 is collected** -> Include rank in every structured event and require the harness to aggregate all worker logs before passing a group.
- **Small 16-request performance samples are noisy** -> Use paired fresh services and identical input/order for the gate, retain the 85% floor, and use separate profiles for causal performance conclusions.

## Migration Plan

1. Create an implementation branch/worktree from `959b9a6a` without changing the frozen vLLM checkout.
2. Implement the exact-scope policy and tests before enabling any FULL-specific adaptation.
3. Build/install only the vLLM Ascend plugin artifact required by the existing environment and record its source/package identity.
4. Run staged smoke and formal validation on explicitly allocated idle cards, preserving all required serving flags.
5. Archive commands, manifests, logs, snapshots, token outputs, benchmark JSON, memory data, comparison tables, and profile artifacts in one dated result root.
6. For rollback, stop only the service started by the acceptance harness, restore the previous plugin commit/package, and use the already-validated eager or Piecewise command. No model, KV-cache data, upstream vLLM, or operator migration is required.
