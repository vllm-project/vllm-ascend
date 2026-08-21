## Context

See `proposal.md` for motivation and `specs/310p-dflash-fdo-numerical-equivalence/spec.md` for the behavioral contract.

The staged FDO implementation at plugin baseline `959b9a6a` completed genuine target/draft capture and replay for the required 9B and 35B matrix. The archived baseline reproduced a stable multi-request cluster and supplied the diagnostic seed. Boundary and layer probes then separated one plugin defect from residual target-model floating-point nondeterminism.

The proven plugin defect was draft context RoPE truncation. During FDO, the query graph descriptor can be smaller than the live context extent. Preparing context RoPE with only the query descriptor omitted valid context rows and changed downstream draft/target decisions. The repair uses `max(query_descriptor_tokens, context_actual_tokens)` for the context descriptor while retaining the live actual count and stable preallocated buffers. No W8A8, MoE event, upstream vLLM, operator, or model-weight change was required.

Post-repair deep tracing found a residual 9B target difference first at layer 1 GDN core. For matched inputs, Eager-to-FDO maximum absolute differences were approximately `1.9e-5` to `3.1e-5`; independent Eager-to-Eager runs reached approximately `3.1e-5` to `3.8e-5` at the same boundary. Because the control mode varies by an equal or larger amount, this is numerical nondeterminism rather than evidence of an FDO-specific computation error.

## Goals / Non-Goals

**Goals:**

- Establish the first matched speculative iteration and the first target/draft numerical boundary that differs.
- Identify the first differing transformer layer and whether the divergence begins in target or draft execution.
- Retain a default-off DEBUG probe capable of observing genuine graph replay without changing normal execution.
- Prove one root-cause hypothesis at a time and implement one minimal plugin-side correction.
- Close the 35B multi-request divergence while preserving every established FDO graph, safety, quality, performance, and memory gate.

**Non-Goals:**

- Treating general FP16 non-associativity as sufficient explanation without locating the responsible path difference.
- Requiring every intermediate floating-point value to be bitwise identical when selected tokens and the parent acceptance contract are satisfied.
- Modifying kernels, upstream vLLM, model weights, quantization parameters, or public serving APIs.
- Optimizing synchronization, graph memory, or throughput during correctness diagnosis.
- Investigating a single non-repeating wording-only branch unless it becomes reproducible or part of a multi-request cluster.

## Decisions

### 1. Preserve the staged FDO implementation and diagnose in a separate OpenSpec

The existing staged diff remains the control implementation; this change adds only diagnostic capability and the eventual evidence-backed repair. No commit or push occurs until the parent FDO acceptance and this numerical-equivalence change are resolved.

Alternative considered: fold new experiments directly into the parent FDO tasks. Rejected because the parent contains 49 staged files and formal evidence; a separate change makes every new probe, hypothesis, and acceptance result auditable without rewriting the implementation history.

### 2. Start with one stable C1 prompt, not the full 16-request matrix

The first hardware gate uses Qwen3.6-35B-A3B-w8a8 TP2, GSM8K record 12, concurrency 1, output 64, temperature 0, ignore EOS, DFlash K=15, and FDO capture sizes `[64,32,16]`. Fresh Eager and FDO services must reproduce the archived token-48 branch before instrumentation is trusted. Once repaired, output 256 and the complete matrix return as acceptance gates.

Alternative considered: instrument all 16 requests at concurrency 4. Rejected because concurrency changes descriptor padding and interleaves request lifecycles, making trace alignment and artifact volume unnecessarily difficult before the first boundary is known.

### 3. Align traces by logical execution identity

Each record is keyed by model, mode, component (`target` or `draft`), TP rank, prompt/dataset index, generated-prefix token IDs, speculative iteration, draft substep, descriptor, actual token count, active-row mapping, and semantic tensor role. Comparison refuses to proceed when the identity or active-row mapping differs.

This separates numerical divergence from scheduler divergence. If inputs or identities first differ, investigation moves to the producer of those values instead of inspecting model layers.

### 4. Use a boundary ladder before layer hooks

The initial ladder is:

`input IDs/positions/active metadata -> target final hidden -> target logits -> draft final hidden -> draft logits/proposed IDs -> rejection result`

Only the first boundary transition from match to mismatch is pursued. For logits, the artifact records top-k IDs/values, each side's chosen-token logits, and argmax margins. For hidden states, bounded active rows are saved so the offline comparator can compute exact nonzero counts and numerical distances.

Alternative considered: install hooks on every transformer layer immediately. Rejected because it changes graph size substantially, consumes avoidable NPU memory, and makes it harder to distinguish a model divergence from a scheduler or sampler divergence.

### 5. Make one selected layer a graph output side channel

The opt-in probe allocates persistent device storage before FDO capture and installs exactly one selected target or draft layer hook. During capture, the hook adds a device-to-device copy of the selected active tensor to the graph. Each replay updates the same storage; after the existing replay/consumer ordering completes, DEBUG code copies the bounded active portion to CPU and writes an artifact.

Eager uses the same selected-layer hook and artifact schema. Dummy capture events are marked and excluded from paired runtime comparison. The hook is absent when probing is disabled, so production behavior and memory are unchanged.

Alternative considered: call `torch.save` or CPU conversion from inside the layer hook. Rejected because Python inside capture runs only while the graph is built, introduces synchronization, and would not observe subsequent replay values.

### 6. Locate the first layer by binary search plus adjacency proof

After a model-internal divergence is established, target and draft are searched independently. Each service selects one checkpoint layer. A midpoint match discards the earlier half; a midpoint mismatch retains it. After convergence, adjacent layers N-1 and N are rerun for the same prompt/iteration, and the first-layer claim requires N-1 to match and N to differ.

The default probe selects one layer. An explicit bounded ordered layer set (including `all` for a short diagnostic sweep) is allowed when per-run record/byte limits remain active; this permits first-difference localization without turning the permanent default-off probe into unbounded instrumentation. A final-norm/lm-head boundary is treated as an additional checkpoint when all transformer layers match.

### 7. Compare values with both exact and decision-oriented metrics

Hidden artifacts report exact unequal elements, max/mean absolute difference, relative difference, finite status, and cosine similarity. Logit artifacts additionally report top-k overlap, selected IDs, selected-token cross-evaluation, and argmax margins. Exact equality locates the earliest arithmetic path difference; decision metrics show whether that difference can change greedy sampling.

The repair objective is removal of the reproducible multi-request token cluster, not an unsupported claim that every graph kernel must be bitwise identical to Eager.

### 8. Test W8A8 chunking first only if the first layer supports it

If the earliest difference appears at or immediately after a W8A8 linear projection, the first single-variable experiment makes FDO use the Eager token-dimension matmul partition for the small static FULL descriptor while retaining the existing Piecewise/dynamic large-token safeguard. The experiment is accepted only if it moves/removes the first-layer difference and preserves FDO capture/replay.

If it does not, the experiment is removed before any MoE experiment. If the first difference instead appears at a MoE boundary, the event/ordering policy is investigated first. This ordering is evidence-driven; the candidate list does not authorize either repair in advance.

### 9. Treat MoE event changes as an ordering hypothesis, not a numerical toggle

If the first divergence is a MoE layer, artifacts compare routed expert IDs/weights, shared/routed outputs, and pre/post-combine hidden states for the selected layer. The experiment changes only the relevant event/stream ordering while retaining graph capture. A change that requires an eager island or unsupported event capture is rejected and reported as a scope blocker.

### 10. Use TDD for the probe and for the final repair

Tests are written and observed failing before production changes. Probe tests cover exact-scope activation, default-off zero overhead, active-lane slicing, replay identity, persistent-buffer reuse, dummy-capture exclusion, artifact bounds, and paired comparison. The repair test names the proven root-cause behavior and fails against the unmodified implementation; tests that merely assert source text or a flag value are not accepted.

Hardware evidence supplies the integration RED for the deterministic record-12 branch. A candidate must turn that RED green before broader tests run.

### 11. Keep diagnostic artifacts separate from formal acceptance

Artifacts are stored under a new timestamped `/home/whn/aisbench_runs/fdo_logit_diag_*` root with a manifest containing source identity, launch configuration, probe bounds, and hashes. They are never staged. Performance numbers from probe-enabled runs are diagnostic only because tensor export synchronizes the device.

Formal throughput, acceptance, and memory are measured again with probes disabled.

## Apply Results (2026-08-20)

### Proven first divergence and repair

The aligned ladder established matching scheduler inputs, query IDs/positions, and query RoPE inputs before the first failing draft context boundary. The defect occurred when `context_actual_tokens` exceeded the query descriptor: the FDO RoPE preparation sliced `_context_positions_buffer` to the query descriptor and omitted live context rows. The regression test names this boundary and requires all 64 live context positions to be prepared even when the query descriptor is 16.

The minimal plugin-only repair is in `vllm_ascend/_310p/spec_decode/llm_base_proposer_310.py`. It computes the context descriptor independently as the maximum of the query descriptor and live context extent. The query graph descriptor, routing, graph ownership, operators, quantization, upstream vLLM, and non-FDO paths are unchanged.

The retained DEBUG probe now also receives the live context row count after replay. Python hooks do not execute on every graph replay, so their capture-time row count can be stale; artifact export slices only the runtime rows. This is a default-off observability correction and does not change model execution.

All probe environment variables are registered and documented in `vllm_ascend/envs.py`; the probe resolves central lazy values once during runner construction. The output-directory switch remains unset by default, so probes allocate no buffers and create no files in ordinary serving.

### Residual numerical classification

Post-repair 9B batch-4 probes aligned target input IDs, positions, sample indices, layer-0 output, and layer-1 input. The first difference was layer-1 linear-attention GDN core. Stable Eager-to-FDO comparisons reported maximum absolute differences from about `1.907e-5` to `3.052e-5`; independent Eager-to-Eager runs reported about `3.052e-5` to `3.815e-5`. No FDO-only repair hypothesis is supported when the control path varies by an equal or larger amount.

The fresh 35B TP2 C4 acceptance pair initially reported five wording branches. Eager C1/C4 and an independent C4 Eager repeat proved request 0 at token 165 was Eager baseline drift. Two independent FDO C4 runs had changing branch sets; only request 6 at token 140 remained identical between FDO runs and both Eager controls. Under the user-approved isolated wording-branch policy, that one residual branch is recorded as a scoped numerical limitation and does not authorize a runtime change.

### Formal probes-disabled acceptance

All groups used GSM8K records 0-15, output length 256, temperature 0, ignore EOS, DFlash K=15, and only NPU devices 0/1. Every request completed and every FDO group met the accepted-length, acceptance-rate, throughput, graph-authenticity, and safety thresholds.

| Group | Eager output tok/s | FDO output tok/s | Ratio | FDO accepted length | Token result |
|---|---:|---:|---:|---:|---|
| 9B TP1 C1 | 27.748 | 31.482 | 1.135 | 6.6230 | exact |
| 9B TP1 C4 | 82.382 | 83.194 | 1.010 | 6.5864 | residual branches classified by layer/Eager baseline |
| 9B TP1 C10 | 97.598 | 99.846 | 1.023 | 6.6957 | no crash; residual branches classified by repeat/layer evidence |
| 9B TP2 C1 | 28.665 | 46.010 | 1.605 | 6.6115 | exact |
| 9B TP2 C4 | 96.713 | 113.178 | 1.170 | 6.5204 | exact |
| 35B TP2 C1 | 19.135 | 41.923 | 2.191 | 6.9698 | one Eager-repeat branch plus one isolated FDO branch |
| 35B TP2 C4 | 50.896 | 82.941 | 1.630 | 6.7198 | one stable isolated branch after Eager/FDO repeats |

Formal artifacts are under `/home/whn/aisbench_runs/fdo_context_extent_acceptance_20260820`, `/home/whn/aisbench_runs/fdo_context_extent_acceptance_repeat_20260820`, `/home/whn/aisbench_runs/fdo_final_acceptance_apply2_20260820`, and `/home/whn/aisbench_runs/fdo_final_acceptance_repeat_20260820`. Deep numerical artifacts are under `/home/whn/aisbench_runs/fdo_9b_batch4_target_layer1_gdn4_20260820_eager1`, `..._eager2`, and `..._fdo1`.

## Risks / Trade-offs

- [Probe copy changes graph scheduling enough to hide the bug] -> Copy only one bounded tensor, reproduce the final token branch with the probe enabled, and compare graph manifests/counters before trusting its values.
- [Separate services do not reach the same speculative iteration] -> Key records by generated prefix and input identity; stop at the first alignment mismatch rather than comparing unrelated steps.
- [Full tensor artifacts consume disk or expose prompts] -> Limit to one frozen diagnostic prompt, active rows, selected iterations, owner-controlled directory, and explicit size/iteration caps.
- [A tiny numerical difference is inherent to a different legal kernel] -> Use argmax margins and repeatability to distinguish harmless drift from decision-changing divergence; do not weaken the multiple-branch gate without explicit user approval.
- [A TP rank appears equal while another diverges] -> Retain per-rank artifacts and require both ranks in each paired checkpoint.
- [Candidate repair restores tokens but harms graph authenticity or quality] -> Run graph proof and parent acceptance thresholds after every candidate; reject token-only wins.
- [The safe fix lies in upstream/operator code] -> Stop with the earliest layer/operator evidence and reopen scope; do not patch frozen components.

## Migration Plan

1. Add and validate the default-off probe and paired comparator without changing model behavior.
2. Reproduce the archived 35B record-12 branch and locate the first boundary/layer.
3. Apply one evidence-backed candidate through RED-GREEN tests and the fast hardware gate.
4. Run the full probes-disabled 9B/35B acceptance matrix and update both OpenSpec evidence sets.
5. If any gate regresses, remove only the candidate repair while retaining the diagnostic capability and artifacts for the next hypothesis.
