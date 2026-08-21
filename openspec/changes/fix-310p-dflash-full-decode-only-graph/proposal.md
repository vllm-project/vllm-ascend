## Why

On Ascend 310P, DFlash configured with `CUDAGraphMode.FULL_DECODE_ONLY` does not yet have a verified plugin-side contract proving that uniform decode executes real FULL ACL graphs for both the target and draft models while prefill and mixed batches remain correctly eager. Historical experiments alternated between apparently successful capture/replay and failures inside FULL capture, so this mode must be rebuilt from the current verified Piecewise baseline with strict scope, safety, and evidence instead of restoring reverted graph code.

## What Changes

- Add a strictly gated `310P + DFlash + FULL_DECODE_ONLY` execution path in the vLLM Ascend plugin without changing upstream vLLM.
- Preserve the native `(FULL, NONE)` semantics: uniform DFlash decode is eligible for FULL, while prefill, chunked prefill, prefix-cache transitions, and mixed prefill/decode batches use `NONE`.
- Preserve user-supplied capture sizes and validate `[160,16]` for the final K=15 concurrency matrix.
- Add a focused Qwen3.5-9B TP1 concurrency-10 closure using descriptor 160 without weakening the existing concurrency-1/concurrency-4 correctness, acceptance-length, output, or throughput baselines.
- Reopen the 35B W8A8 MoE concurrency-10 gate for TP2 and TP4: descriptor 160 MUST survive repeated real replay for 20/20 requests without the observed `QuantBatchMatmulV3` AICore/L0C failure, while paired Eager and Piecewise controls remain unchanged.
- Require startup-time FULL capture and runtime replay for both target and DFlash draft components on every participating TP rank; an eager-only component or log-only mode declaration is not a successful graph startup.
- Reuse only verified generic Piecewise foundations such as persistent buffers, recursive graph-input contracts, guarded FX compatibility, W8A8 graph safety, and acceptance tooling behind a new independent FULL_DECODE_ONLY activation predicate.
- Add permanent, default-off DEBUG evidence for requested/resolved/runtime modes, mode transitions, descriptors, capture/replay counts, graph-input addresses, alignment, stream ordering, and classified fallbacks.
- Retain conservative stream synchronization for the first functional repair and profile it only after correctness and graph authenticity pass.
- Add staged unit, hardware, correctness, acceptance, memory, and performance validation for Qwen3.5-4B TP1/TP2 and Qwen3.6-35B-A3B-w8a8 TP2/TP4 at concurrency 1 and 10, using 4 and 20 requests respectively.
- Retain the completed 9B development gates as historical evidence while making the final 4B/35B matrix reproducible from the repository acceptance tool.
- Keep `FULL` and `FULL_AND_PIECEWISE` outside this change.

## Capabilities

### New Capabilities

- `310p-dflash-full-decode-only-graph`: Defines activation, dispatch, capture/replay, safety, observability, compatibility, and acceptance requirements for real DFlash FULL_DECODE_ONLY execution on Ascend 310P.

### Modified Capabilities

None. No existing repository capability specifies 310P DFlash FULL_DECODE_ONLY behavior.

## Impact

- Expected implementation areas are the 310P model runner, DFlash proposer integration, ACL graph wrapper/diagnostics, graph-persistent input handling, and focused unit/e2e acceptance tooling.
- The implementation baseline is vLLM Ascend commit `959b9a6a`; reverted FULL-family commits are evidence sources only and will not be cherry-picked.
- The vLLM 0.24.0 source checkout at `ee0da84a`, torch-npu 2.10.0.post2, and the current 25.5.0 driver/CANN stack remain frozen and unmodified.
- No C++, AscendC, custom-operator, or public API changes are authorized. If an existing operator cannot participate in a genuine FULL graph through a safe plugin-side contract, implementation stops for separate review rather than introducing an eager island or claiming false success.
- Eager, Piecewise, non-DFlash speculative decoding, non-speculative decoding, other graph modes, and non-310P platforms must retain existing behavior.
