# Speculative Decoding Upstream Alignment

## Purpose

vLLM-Ascend should use vLLM as the owner of the generic speculative-decoding
control flow. Ascend code should add platform behavior only where the generic
path cannot express NPU requirements.

This document records the audited ownership boundary and the regression gates
that keep the two runners aligned. It is based on:

- vLLM-Ascend `main` at `128f35ed1461ecb34bc8b87583a67a66bc4bd1d6`
- vLLM `main` at `a98247ab4db686ee03c66d5feb3c761e52a2f8ab`

The revisions must be refreshed whenever the supported vLLM baseline changes.

## Unification Contract

The default rule is:

1. vLLM owns draft input preparation, multi-step draft iteration, sampling,
   rejection or verification, accepted-token accounting, and output restore.
2. vLLM-Ascend subclasses or adapts those paths before or after `super()`.
3. Ascend-specific attention metadata, NPU buffers, ACL graph capture,
   PCP/DCP layout, and NPU operators remain in vLLM-Ascend behind narrow hooks.
4. A platform patch is retained only while no accepted upstream extension point
   can express the same behavior.
5. A new speculative family is added to the shared selector before either
   runner grows method-specific dispatch.

The two runners must select the same speculative method family even when their
implementation classes differ. `vllm_ascend.spec_decode.method` owns that
runner-independent classification. MRV1 dispatch, MRV2 dispatch, the 310P MRV2
router, and the MRV2 readiness gate all consume that result.

## Current Boundary

| Path | Shared or upstream owner | Ascend-specific boundary | Current action |
| --- | --- | --- | --- |
| Family selection | `vllm_ascend.spec_decode.method` | N/A | Shared by MRV1, MRV2, 310P routing, and MRV2 readiness checks. |
| MRV1 proposer lifecycle | `vllm/v1/spec_decode/*` | NPU model loading, attention metadata, ACL graph buffers, PCP/DCP coordination | Generic proposer and sampling methods delegate to upstream where their interfaces match. |
| MRV1 DFlash | Upstream DFlash proposer | NPU copy/expand kernel and persistent slot-mapping buffers | Model-specific input preparation stays behind narrow overrides. |
| MRV1 DSpark | Upstream DSpark proposer interfaces | NPU attention metadata, multi-group KV layout, and graph updates | Sampling and sequential draft generation remain upstream-owned. |
| MRV2 autoregressive | Upstream `TargetDependentARSpeculator` propose loop | Ascend metadata construction, replicated PCP, graph capture, and hidden-state broadcast | Overrides wrap upstream calls instead of reimplementing the control flow. |
| MRV2 multi-KV-group MTP | Upstream `MTPSpeculator` lifecycle | Ascend model loading, metadata, graph handling, and LM-head TP alignment | Shares the generic Ascend MTP speculator with regular MTP; MRV1 keeps its dedicated proposer. |
| MRV2 DSpark | Upstream DSpark sampling and sequential decode | Ascend model loading, LM-head TP alignment, attention metadata, and graph handling | Draft sampling remains inherited from upstream. |
| Rejection sampling | Upstream rejection-sampler host orchestration and portable helpers | Ascend Triton kernel and NPU launch constraints | Retain only the kernel implementation that cannot run upstream on NPU. |
| KV cache metadata | vLLM block-table and slot-mapping interfaces | PCP/DCP ownership mapping, ACL graph padding, and reusable NPU buffers | MTP, Step3.5, and multi-KV-group paths share one slot-mapping write helper. |
| Method patches | vLLM runner modules | `vllm_ascend/patch/worker/patch_v2/*` | Retain only while the verified upstream revision lacks an extension point. |

The slot-mapping write contract is centralized in
`vllm_ascend.spec_decode.utils.copy_slot_mapping_to_padded_buffer`. It owns
dtype conversion, source aliasing, and graph-padding sentinel values for MTP,
Step3.5 MTP, and multi-KV-group MTP.

MRV2 readiness falls back to MRV1 for speculative families without an Ascend
V2 speculator: suffix, NGram, Medusa, standalone draft model, Gemma4 MTP, and
Step3.5 MTP. DFlash2 and parallel-drafting EAGLE are excluded only when their
documented graph limitations apply.

## Difference Inventory

The audit separates portable algorithm logic from Ascend execution details. The
following inventory is the hand-off boundary for reducing duplicate code without
removing NPU behavior that upstream cannot express.

| Area | Current Ascend owner | Upstream owner | Classification | Migration action |
| --- | --- | --- | --- | --- |
| Speculative family selection | `vllm_ascend/spec_decode/method.py` | vLLM `SpeculativeConfig` method/use helpers | Portable policy | Keep one Ascend selector for runner parity; add new families here before adding runner branches. |
| MRV1 proposer lifecycle | `AscendSpecDecodeBaseProposer` | `vllm/v1/spec_decode/llm_base_proposer.py` | Mixed | Reuse upstream sampling, proposal, and accounting methods. Retain overrides only for Ascend metadata, graph, PCP/DCP, and custom operators, then narrow those overrides behind hooks. |
| MRV1 DFlash/DSpark input preparation | `AscendDflashProposer`, `AscendDSparkProposer` | `vllm/v1/spec_decode/dflash.py` and base proposer | Mostly NPU | Retain NPU copy/expand and graph-safe buffers; delete any duplicated host-side preparation already supplied upstream. |
| MRV2 autoregressive draft loop | `AscendAutoRegressiveSpeculator` | `TargetDependentARSpeculator` | Mixed | Continue delegating prefill, multi-step decode, and draft iteration to `super()`; keep only metadata construction, ACL Graph, replicated PCP, and hidden-state broadcast overrides. |
| MRV2 DFlash/DFlash2 | `AscendDFlashSpeculator`, `AscendDFlash2Speculator` | Upstream DFlash speculators | Mostly NPU | Retain NPU input-preparation kernels and graph managers; portable sampling and sequential decode stay upstream. |
| MRV2 DSpark | `AscendDSparkSpeculator` | Upstream `DSparkSpeculator` | Mixed | Inherit upstream draft sampling; retain only LM-head TP alignment, attention metadata, PCP/DCP, and graph handling. |
| Draft sampling | Shared proposer/speculator wording | Upstream `_sample_draft_tokens`, `compute_probs_and_sample_next_token`, and speculator samplers | Portable algorithm | Use upstream sampling. Ascend-only LM-head TP padding and fixed graph buffers remain narrow mixins. |
| Rejection sampling | `worker/v2/spec_decode/rejection_sampler_utils.py` | GPU rejection sampler host orchestration/helpers | Kernel-specific copy | Keep the Ascend Triton kernel and launch constraints. Re-import portable helpers instead of maintaining a second host algorithm. |
| KV cache metadata and slot mapping | Ascend proposer/speculator adapters and `spec_decode/utils.py` | vLLM block-table/slot-mapping interfaces | Mixed | Use vLLM metadata inputs; centralize NPU dtype, alias, graph-padding, PCP/DCP, and reusable-buffer details in one helper. |
| Runtime patches | `vllm_ascend/patch/worker/patch_v2/*` | vLLM runner/speculator modules | Temporary compatibility | Retain only for revisions without an extension point. Record removal condition and upstream issue/PR for each patch. |

The inventory deliberately does not label an override as duplicated solely
because it has the same method name. An override is removable only when the
upstream implementation can consume Ascend metadata and graph buffers through an
existing extension point.

## Migration Order

1. **Shared classification and readiness.** Complete family selection and make
   MRV1, MRV2, 310P routing, and V2 fallback consume the same result.
2. **Portable sampling and rejection helpers.** Replace copied host algorithms
   with upstream imports. Keep only NPU kernels, fixed-shape constraints, and
   graph-safe wrappers.
3. **MRV1 proposer hook reduction.** Move portable proposal and accounting logic
   back behind `super()` calls. Keep Ascend overrides for metadata, graph, and
   parallel-layout work.
4. **MRV2 speculator hook reduction.** Apply the same process to autoregressive,
   DFlash, and DSpark speculators. Preserve PCP/DCP and LM-head TP behavior in
   mixins or narrow methods.
5. **Patch retirement.** Remove a patch after the supported vLLM baseline exposes
   the required extension point and the family-specific E2E case passes on the
   release and main2main lanes.

Each migration step must leave the previous configuration usable. No step may
change the default runner selection without a documented fallback.

## Compatibility And Rollback

- No speculative-config or CLI surface is changed by the alignment refactor.
- `VLLM_USE_V2_MODEL_RUNNER=0` remains the explicit rollback to MRV1.
- Families without an Ascend V2 speculator continue to fall back to MRV1.
- If a family-specific V2 regression is found, remove that family from the V2
  supported set and restore the previous V1 route. Do not revert unrelated
  families or shared slot-mapping fixes.
- Reverting a migration commit must restore the previous dispatch condition,
  patch, and tests together; mixing old dispatch with new family classification
  is unsupported.
- New patches must include an upstream issue/PR or explain why no upstream
  extension point can express the behavior.

## NPU-Only Adaptations

The following deviations are intentional and must remain covered by tests:

- Ascend attention metadata builders and attention-state selection.
- ACL graph capture, graph parameter updates, padding, and replay.
- PCP/DCP layout conversion and context-parallel request staging.
- NPU custom operators and Triton kernels that upstream cannot lower on Ascend.
- LM-head tensor-parallel row alignment where Ascend uses fixed-size buffers.
- Model-family compatibility shims that have no generic upstream equivalent.

An adaptation must be narrow. If it changes draft input preparation, sampling,
rejection, or accepted-token accounting for every platform, it belongs upstream
instead of in a subclass or patch.

## Validation

Focused CPU tests verify family selection, MTP precedence, agreement across all
dispatch and readiness entry points, slot-mapping behavior, inheritance, hook
invocation, and delegation to upstream-owned sampling and processor paths. The
speculative-decoding-focused command used for this change is:

```bash
python -m pytest -sv \
  tests/ut/spec_decode/test_method.py \
  tests/ut/test_mrv2_utils.py \
  tests/ut/worker/v2/test_spec_decode.py \
  tests/ut/sample/test_rejection_sampler.py \
  tests/ut/_310p/spec_decode/test_mtp_mrv2_310.py \
  tests/ut/spec_decode/test_utils.py
```

It covers proposer/speculator assembly, malformed and unsupported configuration,
greedy acceptance, non-greedy acceptance and rejection, placeholder rejection,
per-position acceptance accounting, and 310P MRV2 behavior.

NPU validation should cover the affected execution mode:

- eager, PIECEWISE, and FULL graph capture and replay where supported;
- MTP and EAGLE3 for MRV2 changes;
- DFlash, DSpark, or the affected draft family when its path changes;
- applicable TP, DP, PP, PCP, and DCP combinations;
- output correctness and draft acceptance against the pre-change baseline;
- no material latency or throughput regression.

Current hardware verification uses an Ascend A3 device:

- MTP with `Qwen3.5-4B` passed eager and `FULL_DECODE_ONLY` execution in MRV2.
- EAGLE3 with `Qwen3-0.6B` and the public SpecForge draft checkpoint passed the
  same two execution paths in MRV2.
- The public draft checkpoints are configurable through
  `SPEC_DECODE_MTP_MODEL`, `SPEC_DECODE_EAGLE3_MODEL`, and the corresponding
  `*_DRAFT_MODEL` variables.

Model-backed DSpark requires the 27B target and exceeds the memory of the
single-NPU development environment. Its two-card nightly case is registered
separately and is not claimed as locally executed by this change. DFlash remains
family-specific CI coverage. CPU-only unit tests are not acceptance evidence for
NPU attention layouts or graph replay.

The nightly refactor gate runs the same fixed prompts against the target model and
the speculative family in eager and `FULL_DECODE_ONLY` modes. Both paths must
produce non-empty output and speculative acceptance above one. Cache block size,
cache dtype, prefix caching, and Mamba cache mode must match. The number of
available KV blocks is recorded but is not required to match across modes because
Graph capture changes the remaining HBM budget. Eager and Graph are independent
execution modes and are not required to produce the same long-horizon greedy
tokens; ACL Graph numerical drift on the audited MTP model was observed after
token 20.

The baseline/candidate benchmark compares the same family in the same execution
mode. Greedy token IDs over the fixed parity window, cache signature (including
KV block count), acceptance length, and per-position acceptance must match within
the documented tolerances. This separates a control-flow regression from normal
eager-versus-Graph numerical differences.

Performance is measured with identical model, prompts, token limits, TP/EP,
graph mode, seed, and memory utilization before and after the refactor. Record
throughput, TTFT, decode latency/ITL, and peak HBM. Any regression must be
reproduced and explained; an unexplained slowdown above 3% blocks acceptance.
Correctness parity takes precedence over performance when the two conflict.

The offline comparison is reproducible from each checkout with separate compile
caches:

```bash
VLLM_CACHE_ROOT=/tmp/vllm-cache-baseline \
python -m tests.e2e.nightly.feature.spec_decode_refactor_benchmark record \
  --family mtp --mode full_decode_only --output /tmp/base-mtp.json

VLLM_CACHE_ROOT=/tmp/vllm-cache-candidate \
python -m tests.e2e.nightly.feature.spec_decode_refactor_benchmark record \
  --family mtp --mode full_decode_only --output /tmp/candidate-mtp.json

python -m tests.e2e.nightly.feature.spec_decode_refactor_benchmark compare \
  /tmp/base-mtp.json /tmp/candidate-mtp.json
```

For TTFT and ITL, start each checkout with the same `vllm serve` arguments and
run `vllm bench serve --temperature 0 --save-result`. Pass that result JSON to
the record command with `--serving-json` when both runs are being recorded.

The local A3 comparison used baseline `e04c70f` and the candidate branch on
`Qwen3.5-4B` MTP with three draft tokens and a 16-token parity window. Separate
compile caches were used for each checkout to avoid cross-worktree AOT cache
reuse.

| Mode | Tokens | Acceptance | Throughput (tok/s) | Latency (s) | HBM (bytes) |
| --- | --- | --- | --- | --- | --- |
| eager baseline | exact | 4.0000 | 35.57 | 1.7994 | 55,606,603,776 |
| eager candidate | exact | 4.0000 | 34.70 | 1.8445 | 55,606,743,040 |
| Graph baseline | exact | 4.0000 | 39.69 | 1.6126 | 65,080,582,144 |
| Graph candidate | exact | 4.0000 | 39.33 | 1.6271 | 65,081,389,056 |

The eager change is a 2.44% throughput and 2.50% latency delta, inside the 3%
gate. Graph changes are 0.90% throughput, 0.90% latency, and 0.001% HBM. The
per-position acceptance rates were `[1.0, 1.0, 1.0]` for all four runs and the
cache signatures were identical.

Serving measurements used `vllm bench serve`, eight random prompts, one warmup,
128 input tokens, 16 output tokens, concurrency one, and greedy sampling:

| Metric | Baseline | Candidate | Delta |
| --- | --- | --- | --- |
| Mean TTFT (ms) | 158.20 | 160.82 | +1.66% |
| Mean ITL / decode latency (ms) | 92.25 | 92.42 | +0.19% |
| Mean end-to-end latency (ms) | 780.86 | 784.67 | +0.49% |
| Output throughput (tok/s) | 20.48 | 20.38 | -0.49% |
| Acceptance length | 2.37 | 2.37 | 0% |

EAGLE3 on `Qwen3-0.6B` passed eager/Graph execution with acceptance length 2.10
on both paths. Its local generation rates were 32.68 tok/s eager and 61.06 tok/s
Graph. The development environment has one NPU, so the two-card DSpark case is
left to the registered nightly lane rather than reported as a local pass.
