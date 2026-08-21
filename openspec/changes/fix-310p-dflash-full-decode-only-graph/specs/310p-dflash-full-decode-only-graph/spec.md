## Purpose

Define a verifiable behavioral contract for genuine, safe, and observable DFlash `FULL_DECODE_ONLY` ACL graph execution on Ascend 310P while preserving eager behavior for prefill and mixed batches and leaving every out-of-scope path unchanged.

## ADDED Requirements

### Requirement: Strict activation scope
The system SHALL activate this capability only when the platform is Ascend 310P, the speculative decoding method is DFlash, and the requested and resolved graph mode is `FULL_DECODE_ONLY`. Eager, Piecewise, `FULL`, `FULL_AND_PIECEWISE`, non-DFlash speculative decoding, non-speculative decoding, and non-310P platforms MUST retain their existing behavior.

#### Scenario: Exact scope activates the capability
- **WHEN** an Ascend 310P service uses DFlash with requested and resolved mode `FULL_DECODE_ONLY`
- **THEN** the scoped FULL decode-only behavior is activated

#### Scenario: Other graph modes remain independent
- **WHEN** the requested or resolved mode is `PIECEWISE`, `FULL`, `FULL_AND_PIECEWISE`, or `NONE`
- **THEN** this capability is not activated and no mode is coerced into `FULL_DECODE_ONLY`

### Requirement: Frozen plugin-only compatibility
The capability SHALL use vLLM Ascend commit `959b9a6a` as its implementation baseline with vLLM 0.24.0 at `ee0da84a`, torch-npu 2.10.0.post2, and the current 25.5.0 driver/CANN environment. The vLLM source checkout and C++, AscendC, custom-operator implementations MUST remain unmodified.

#### Scenario: Supported dependency set
- **WHEN** the service runs on the frozen dependency set
- **THEN** FULL decode-only compatibility is supplied entirely by the vLLM Ascend plugin

#### Scenario: Safe plugin adaptation is impossible
- **WHEN** a required existing operation cannot participate in a genuine FULL graph through a safe plugin-side contract
- **THEN** startup or validation fails with the original incompatibility instead of changing the operator, creating an eager island, or claiming graph success

### Requirement: Native FULL and NONE routing
The system SHALL preserve the native `FULL_DECODE_ONLY` runtime split. A batch SHALL be eligible for `FULL` only when all active requests are in DFlash uniform decode with per-request query length `1 + K`, which is 16 for K=15. Prefill, chunked prefill, prefix-cache transitions, and mixed prefill/decode batches SHALL use runtime mode `NONE`.

#### Scenario: Uniform DFlash decode uses FULL
- **WHEN** all active requests are in DFlash uniform decode with query length 16 and the batch maps to a configured descriptor
- **THEN** the target and draft execution use runtime mode `FULL`

#### Scenario: Prefill or mixed work uses NONE
- **WHEN** a batch contains prefill, chunked prefill, a prefix-cache transition, or mixed prefill/decode work
- **THEN** the batch uses runtime mode `NONE` without being classified as an unexpected fallback

#### Scenario: Runtime returns to FULL
- **WHEN** a request transitions from prefill or a cache transition into eligible uniform decode
- **THEN** subsequent eligible decode iterations return from `NONE` to `FULL`

### Requirement: FULL execution has no model eager islands
For eligible uniform decode, FULL execution SHALL include target model forward and attention plus the DFlash draft proposal, draft model forward, draft attention, and required device-side sampling computation. Only persistent-input updates, host scheduling metadata preparation, stream ordering, and output consumption MAY occur outside the graph.

#### Scenario: Genuine end-to-end FULL decode
- **WHEN** an eligible uniform decode descriptor replays
- **THEN** no target or draft model layer or attention operation is silently executed as an eager island

#### Scenario: A graph-incompatible operation is encountered
- **WHEN** an operation inside the required target or draft FULL region cannot be captured
- **THEN** capture fails with the operation and descriptor identified rather than excluding the operation from the graph

### Requirement: Complete startup-time capture
Graph startup SHALL be considered successful only after every configured decode descriptor has completed FULL capture for both target and draft components on every participating TP rank and the API health check succeeds. Runtime lazy capture and a log-only mode declaration MUST NOT satisfy startup.

#### Scenario: Capture manifest is complete
- **WHEN** service initialization finishes for configured sizes `[64,32,16]`
- **THEN** the capture manifest contains target and draft FULL captures for descriptors 64, 32, and 16 on every TP rank before graph startup is reported successful

#### Scenario: One capture is missing
- **WHEN** any component, descriptor, or TP rank lacks a required FULL capture
- **THEN** graph startup validation fails and identifies the missing manifest entry

### Requirement: User capture configuration is preserved
The system SHALL preserve the user-supplied FULL decode capture sizes without replacing them with plugin-selected constants. The formal acceptance configuration SHALL use `[64,32,16]`, mapping K=15 concurrency 4, 2, and 1 to descriptors 64, 32, and 16 respectively.

#### Scenario: Formal sizes are honored
- **WHEN** the user supplies `cudagraph_capture_sizes=[64,32,16]`
- **THEN** the resolved configuration and capture manifest report the same active sizes and use their valid descriptor mappings

#### Scenario: Uniform decode cannot map safely
- **WHEN** eligible uniform decode cannot map to a safe configured descriptor
- **THEN** DEBUG and acceptance validation fail with actual tokens, requests, query length, configured sizes, padded descriptor, component, and rank

### Requirement: Target and draft replay on every rank
Every eligible formal run SHALL demonstrate nonzero FULL capture and replay counts independently for the target and DFlash draft components on every participating TP rank. A target-only graph, draft-only graph, or all-eager run MUST fail graph validation.

#### Scenario: Both components replay
- **WHEN** warmup and formal requests repeat an eligible descriptor
- **THEN** every TP rank reports nonzero target FULL replay and nonzero draft FULL replay

#### Scenario: A component remains eager
- **WHEN** either component has zero FULL captures or zero FULL replays after eligible work
- **THEN** graph validation fails and identifies the component, rank, and descriptor

### Requirement: Capture and replay inputs are safe
Every tensor, view, and workspace retained by a target or draft FULL graph SHALL preserve its required device address, base storage, storage offset, accessible bounds, dtype, shape, stride, device, and declared alignment between capture and replay. Unsafe input state MUST be rejected before replay.

#### Scenario: Stable aligned replay
- **WHEN** an eligible descriptor replays after concurrency changes, prefix-cache activity, or request completion
- **THEN** all retained inputs satisfy their captured identity, layout, bounds, and alignment contracts

#### Scenario: Unsafe retained input is detected
- **WHEN** an input address, base storage, view offset, bounds, dtype, shape, stride, device, or required alignment differs from capture
- **THEN** replay is rejected with the tensor role, component, rank, descriptor, captured contract, and observed contract

### Requirement: Dynamic request lifecycle stays current
FULL replay SHALL consume current block tables, slot mappings, sequence lengths, query boundaries, positions, rejection counts, hidden-state inputs, masks, and token-selection metadata after concurrency changes, partial completion, request condense, and prefix-cache hit or miss transitions. Stale request state MUST NOT be replayed.

#### Scenario: Concurrency transitions
- **WHEN** eligible traffic changes from concurrency 1 to 4 and back to 1
- **THEN** descriptor routing and every retained metadata input reflect the current active batch and outputs remain eager-equivalent

#### Scenario: Completion changes the active batch
- **WHEN** one or more requests finish and the active batch is condensed
- **THEN** the next NONE or FULL iteration uses refreshed request indices and cache metadata

### Requirement: Stream ordering is conservative and correct
The first functional implementation SHALL preserve ordering between persistent-input updates, graph-parameter updates, request condense, current-stream work, update-stream work, and FULL replay. Synchronization MAY be reduced only after event-order tests and profiling prove that replay cannot observe stale data.

#### Scenario: Updates precede replay
- **WHEN** host or device metadata is updated for an eligible descriptor
- **THEN** FULL replay begins only after all required updates are visible on the replay stream

#### Scenario: Synchronization optimization is unproven
- **WHEN** no test and profile evidence proves a synchronization point redundant
- **THEN** the synchronization point remains in place

### Requirement: Fallbacks are classified and validation fails closed
Every runtime `NONE` decision or eager fallback SHALL have a closed, machine-parseable reason. Expected prefill and mixed-batch `NONE` routing SHALL remain valid. Unexpected fallback of eligible uniform decode SHALL fail DEBUG and acceptance validation, while INFO operation MAY continue eagerly with a counter and reason. Safety-contract failures MUST fail in every logging mode.

#### Scenario: Expected NONE routing occurs
- **WHEN** the runtime processes declared prefill, mixed, or cache-transition work
- **THEN** diagnostics increment the corresponding expected `NONE` reason without failing validation

#### Scenario: Eligible decode falls back in validation
- **WHEN** eligible uniform decode runs eagerly in DEBUG or acceptance mode
- **THEN** validation fails with the descriptor and classified reason

#### Scenario: Safety validation fails in INFO
- **WHEN** a retained FULL input violates its safety contract while INFO logging is active
- **THEN** replay is rejected rather than silently falling back

### Requirement: Permanent graph observability
The capability SHALL provide default-off DEBUG evidence containing requested, resolved, and runtime modes; FULL/NONE transitions; configured sizes; component; TP rank; descriptor; capture and replay counts; fallback counts by reason; retained-input contracts; alignment results; and stream-order milestones. INFO logging MUST avoid per-step address output and material diagnostic overhead.

#### Scenario: DEBUG evidence is enabled
- **WHEN** an acceptance run uses DEBUG logging
- **THEN** structured logs or snapshots contain sufficient per-component and per-rank evidence to prove genuine FULL decode and expected NONE routing

#### Scenario: INFO logging is used
- **WHEN** DEBUG evidence is disabled
- **THEN** per-step addresses are not emitted and only concise startup, counter, and error information remains

### Requirement: Required serving features stay enabled
The capability SHALL operate with prefix caching, chunked prefill, asynchronous scheduling, DFlash K=15, `enable_npugraph_ex=false`, `fuse_norm_quant=false`, and the agreed model and memory settings. Success MUST NOT be obtained by disabling these features, lowering K, changing graph mode, or changing compilation backend.

#### Scenario: Complete serving configuration is used
- **WHEN** the agreed deployment command is launched
- **THEN** the service completes genuine FULL decode capture/replay without altering the required flags

### Requirement: Deterministic output matches eager or is numerically classified
For identical frozen software, models, prompts, order, and `temperature=0` sampling, FULL_DECODE_ONLY output SHOULD match the paired DFlash eager baseline token for token. A mismatch is non-blocking only when repeated Eager and FULL_DECODE_ONLY runs plus layer-level evidence classify it as isolated floating-point branch drift, while request success, graph authenticity, safety, acceptance quality, and throughput all pass. Multiple stable requests branching at the same token positions MUST remain blocking until localized and repaired or shown by repeat controls to be run-to-run numerical drift.

#### Scenario: Tokens match
- **WHEN** eager and FULL_DECODE_ONLY process the same formal GSM8K request
- **THEN** their generated token ID sequences are identical

#### Scenario: A token differs
- **WHEN** a generated token differs from the eager baseline
- **THEN** validation records the request index and first differing token position, repeats both controls, and either fails on a stable multi-request branch cluster or records a single isolated numerically classified branch as non-blocking

### Requirement: Quantitative acceptance thresholds
Every final concurrency-1 group SHALL complete 4 of 4 requests and every final concurrency-10 group SHALL complete 20 of 20 requests. FULL_DECODE_ONLY average acceptance length SHALL be at least 5.0 and at least 90% of its paired eager value. Draft-token acceptance rate SHALL be no more than 5 percentage points below eager. Output-token throughput SHALL be at least 85% of paired eager throughput.

#### Scenario: Thresholds pass
- **WHEN** success, output, acceptance, and throughput satisfy every threshold
- **THEN** the formal group passes quantitative acceptance

#### Scenario: A threshold fails
- **WHEN** any quantitative threshold is violated
- **THEN** the formal group fails and reports eager, FULL_DECODE_ONLY, and available Piecewise reference values side by side

### Requirement: Concurrency-10 repair preserves healthy 9B controls
A repair intended to make Qwen3.5-9B TP1 concurrency 10 complete SHALL preserve active-lane draft semantics, deterministic output, mean acceptance length, and throughput for the already-working concurrency-1 and concurrency-4 controls. A request-success improvement SHALL NOT pass when it materially lowers draft quality or throughput. The authoritative regression baseline SHALL be freshly measured after removing only the concurrency-10 repair delta under the same 256-token test configuration; the historical healthy results of mean acceptance length 7.4833 at concurrency 1 and 7.4755 at concurrency 4 remain directional evidence.

#### Scenario: Candidate enters the fast gate
- **WHEN** a concurrency-10 candidate is ready for hardware evaluation
- **THEN** Qwen3.5-9B TP1 first runs GSM8K records 0-3 with output length 256 at concurrency 1 and concurrency 10, and the candidate is rejected on any output mismatch, request failure, graph-contract error, or acceptance-length regression below 90% of the freshly measured control

#### Scenario: Existing concurrency controls remain healthy
- **WHEN** a candidate passes the fast gate
- **THEN** Qwen3.5-9B TP1 runs GSM8K records 0-15 with output length 256 at concurrency 1 and 4, completes 16 of 16 requests, satisfies the deterministic-output classification policy, retains at least 90% of the fresh control acceptance length, and retains at least 85% of the fresh control output throughput

#### Scenario: Concurrency 10 closes without a quality regression
- **WHEN** Qwen3.5-9B TP1 runs GSM8K records 0-15 with output length 256 at concurrency 10
- **THEN** all 16 requests complete, configured FULL target and draft graphs replay through descriptor 160 and tail descriptors, output satisfies the deterministic-output classification policy, mean acceptance length is at least 90% of its paired eager value, and output throughput is at least 85% of its paired eager value

#### Scenario: Tail padding changes an active lane
- **WHEN** descriptor padding or a persistent-input update changes any active-lane position, RoPE value, draft token, or draft probability relative to the unpadded reference
- **THEN** validation fails at the first differing iteration and identifies the active request, token position, tensor role, and descriptor

#### Scenario: Historical incomplete concurrency-10 metrics are inspected
- **WHEN** the pre-repair concurrency-10 run reports mean acceptance length 6.7802 before failing after 11 of 16 requests
- **THEN** that value is retained as diagnostic evidence only and is not treated as a passing or complete baseline

### Requirement: Complete model and parallelism matrix
Final formal acceptance SHALL cover Qwen3.5-4B at TP1 and TP2 plus Qwen3.6-35B-A3B-w8a8 at TP2 and TP4, each at concurrency 1 and 10. Concurrency 1 SHALL use original-order GSM8K records 0-3; concurrency 10 SHALL use records 0-19. Every group SHALL use output length 256, `temperature=0`, `ignore_eos=true`, DFlash K=15, and capture sizes `[160,16]`. Qwen3.6-35B single-card execution is excluded because it cannot be deployed on one card. The earlier 9B C1/C4/C10 and 35B C1/C4 runs remain historical development evidence, not the final reproducible matrix.

#### Scenario: Eight formal groups complete
- **WHEN** the acceptance suite finishes
- **THEN** archived results exist for 4B TP1 C1/C10, 4B TP2 C1/C10, 35B TP2 C1/C10, and 35B TP4 C1/C10

### Requirement: Warmup is explicit and excluded
ACL graph internal warmup SHALL remain `cudagraph_num_of_warmups=0`. Each formal group MAY issue up to four ordinary inference requests after startup to exercise real replay, and those requests MUST be excluded from the 4-request or 20-request latency, throughput, output, and speculative acceptance metrics.

#### Scenario: Ordinary warmup runs
- **WHEN** up to four warmup requests execute before formal measurement
- **THEN** diagnostics preserve their capture/replay effects while formal metrics include only the subsequent 4 or 20 requests

### Requirement: Deployment memory remains viable
Each required model/TP configuration SHALL start with `gpu_memory_utilization=0.85` and the full required capture-size set. Validation SHALL record graph memory and resulting KV-cache capacity without imposing an absolute graph-memory limit. An OOM MUST be addressed by correcting unnecessary graph or buffer ownership rather than weakening the agreed configuration.

#### Scenario: Required deployment starts
- **WHEN** a required model/TP group launches with memory utilization 0.85
- **THEN** startup completes and records graph memory plus final KV-cache capacity

#### Scenario: Graph allocation causes OOM
- **WHEN** required FULL graph capture exhausts memory
- **THEN** the group fails and retains the capture manifest and allocation evidence without deleting required descriptors or disabling serving features

### Requirement: Post-acceptance profiling is non-blocking
After functional acceptance, validation SHALL profile 9B TP1 concurrency 1 and 35B TP2 concurrency 4 under Eager, Piecewise, and FULL_DECODE_ONLY to identify synchronization, host-update, draft-proposal, and replay costs. The first functional change SHALL report these comparisons but SHALL NOT require FULL_DECODE_ONLY to outperform Piecewise.

#### Scenario: Functional acceptance succeeds
- **WHEN** the correctness and authenticity gates have passed
- **THEN** the two three-mode profile comparisons are captured outside formal acceptance measurements

### Requirement: 35B W8A8 concurrency-10 replay is stable
Qwen3.6-35B-A3B-w8a8 with DFlash K=15 SHALL complete the frozen GSM8K concurrency-10 workload in FULL_DECODE_ONLY on TP2 and TP4. Each group SHALL use 20 original-order requests, output length 256, `temperature=0`, `ignore_eos=true`, and capture sizes `[160,16]`. A run passes only with 20/20 successful requests, genuine target and draft FULL replay on every rank, and no `QuantBatchMatmulV3`, L0C conflict, ACL 507015, HCCL ERR02005, graph-contract, or replay error.

#### Scenario: Descriptor 160 is reused
- **WHEN** the second and later uniform-decode batches reuse descriptor 160
- **THEN** every rank completes replay with the current request's activation, routing, quantization, workspace, stream, and lifecycle state rather than state retained from capture or the prior replay

#### Scenario: TP2 gate passes
- **WHEN** the frozen 35B TP2 concurrency-10 workload runs
- **THEN** all 20 requests complete and target plus draft replay descriptor 160 without operator or runtime failure

#### Scenario: TP4 gate passes
- **WHEN** the same frozen workload runs at TP4 after TP2 passes
- **THEN** all 20 requests complete and every rank satisfies the same replay and error checks

### Requirement: Existing modes remain fixed controls
A 35B W8A8 concurrency-10 repair SHALL NOT change Eager or the already-released Piecewise activation, dispatch, graph inputs, operator selection, output, acceptance quality, or throughput. The repair SHALL be exact-scoped to `310P + DFlash + FULL_DECODE_ONLY` unless evidence proves a mode-neutral defect and explicit regressions cover every caller.

#### Scenario: Candidate repair is evaluated
- **WHEN** a candidate passes the focused unit test and TP2 fast gate
- **THEN** paired Eager and Piecewise controls run with identical inputs and retain 20/20 success, acceptance length at least 90% of their frozen baselines, and output throughput at least 85% of their frozen baselines

#### Scenario: Out-of-scope behavior changes
- **WHEN** Eager, Piecewise, non-DFlash, non-speculative, or non-310P behavior changes without an independently proven mode-neutral contract
- **THEN** the candidate is rejected

### Requirement: Ordinary non-DFlash FDO is a diagnostic reference
The investigation SHALL compare the failing DFlash FDO path with ordinary non-DFlash FDO without assuming that every current-environment ordinary-FDO startup failure shares the DFlash concurrency-10 root cause. Historical ordinary 310P FDO support and its request-condense synchronization fix SHALL be preserved.

#### Scenario: Ordinary FDO passes in the matching reference environment
- **WHEN** ordinary FDO completes capture and repeated replay under a source/runtime combination known to support it
- **THEN** its retained-input and lifecycle behavior is used as a reference for the DFlash-specific target/draft boundary

#### Scenario: Ordinary FDO fails before replay for an independent reason
- **WHEN** the current comparison fails during FX size-node decomposition before HTTP readiness
- **THEN** that failure is recorded separately and MUST NOT be used as evidence that the DFlash descriptor-160 operator failure is fixed or reproduced
