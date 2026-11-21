# MoonEP on vLLM Ascend main

## Current main2main baseline

The integration branch `moonep_main` replays the six MoonEP commits through
`moonep_v0.30.0` (`8d173bb61238`) on vllm-ascend main (`4e11513255d8`).
Its verified vLLM revision is `ba15685ab7f4` (main2main 1106).
The working directory is `/mnt/share/x00893896/vllm_ascend_moonep_integration_main`.
Model Runner V2 initializes MoonEP after weight loading. The companion
`ascend-moonep` checkout is `92f06ff4304d`.

Historical smoke runs on A5-118 with `xzx_moonep_main` exercised Qwen3.6-35B-A3B
with BF16 dispatch at EP2 and DeepSeek-V4-Flash with MXFP8 dispatch at EP8. Each
run completed all 16 GSM8K requests and matched 15 reference answers. These
results describe the tested integration revisions.

## Previous v0.30.0 migration and validation

The v0.30.0 branch migrated the final MoonEP implementation from `moonep_v0.23.0`
(`73d2f9e4e21e911ec86682684bc46d862674a595`) onto
`releases/v0.30.0` (`56279e424b0636afe936d011ff65b9c4e6a6c82e`).

The companion vLLM checkout is `releases/v0.30.0`
(`ced6857afa0ea7b2e3f0846a62e1394e90f15607`). The SDK feature baseline is
`ascend-moonep` `urma_dev` (`6de30d77d0f21721bce35a73018160e56fb9d656`).
The validation container uses `urma_dev_vllm_ascend_v0.30.0` with the CANN 9.2
header-layout compatibility commit `5a312f186bc6bf197df223f67d2cd4f7c9b1274d`.
The original migration changed vllm-ascend source only. The SHMEM compatibility
fix described below additionally requires rebuilding the SDK's SHMEM library.

The workspace used for that migration was
`/mnt/share/x00893896/vllm_ascend_moonep_integration_latest`.
The previous workspace and source branch remain separate baselines.

**Status: CPU regression and fresh-process native-loader checks pass.** On A5-118
in `xzx_moonep_latest`, two-rank SHMEM initialization/finalization passes in both
memfabric/SHMEM load orders. Eight-rank eager model smoke tests pass for
Qwen3-30B-A3B-Instruct-2507 with MoonEP BF16 transport, and DeepSeek-V4-Flash
with MoonEP MXFP8 transport and auto/BF16 attention KV. Each model configuration
completes 11 short-decode, long-prefill, concurrent-completion and chat requests,
with memfabric loaded normally in all workers. These are smoke checks, not a
numerical-equivalence, sparse-offload feature, or performance qualification.

## Execution integration

- `AscendRoutedExperts` registers the actual owner of expert Parameters.
  Runtime allocation and weight publication occur after model weight loading
  in Model Runner V2, before the first forward.
- `MoECommType.MOONEP_SHMEM` registers `MoonEPCommImpl` when enabled. The
  hardware policy selects the native backend when MoonEP is off. With
  `MOONEP_KEEP_MC2=1`, eligible native MC2 and Fused MC2 steps are retained.
- MoonEP selects a common capacity across EP ranks and dispatches the actual
  token count through the SDK. Weighted combine uses the source rank's FP32
  Top-K weights and Home/Slot group counts to process valid rows.
- `moonep_weight_manager.py` publishes local expert tensors to SHMEM Home,
  manages shared prefetch Slots, and builds zero-copy Home/Slot MLP views.
  Supported BF16 and W4A8MXFP parameters can use Home as backing storage after
  publication. V2 accounts for the net change in live Torch allocations after
  MoonEP initialization when budgeting the KV cache.
- `SlotMLPLayerView` supplies Home/Slot aliases to native quant-method hooks.
  Replacing only `MoEMlpComputeInput.weights` is insufficient in v0.30 because
  the hooks obtain tensors from `layer`. The view supplies weights and scales
  without swapping live model Parameters on each forward.
- BF16 and W4A8MXFP/MXFP8 use SDK dispatch and fused publication. The
  dispatcher's row counts determine the valid portion of the output shard.
- Layer selection at the MoE call identifies which weights to prefetch.
  When enabled, the side stream starts prefetch after planning. The main stream
  waits before the unified MLP, or between Home GEMM1 and Slot GEMM1 in Home
  overlap mode. Home overlap requires the optional expert cache to be disabled.

MoonEP requires expert parallelism over multiple ranks, contiguous linear
expert placement, eager execution, and supported unsplit expert tensors.
Dynamic EPLB, mix placement, expert bias, split expert lists, and MoE LoRA are
rejected. Home GEMM supports BF16 gated activations from the existing path and
W4A8MXFP with SiLU; other Home backends fail explicitly.
Native Off retains its own EPLB and communication paths.

## MoonEP semantic compatibility

Non-MoonEP execution follows the main baseline. MoonEP reuses its quant-method
GMM and activation-quantization hooks through a read-only Home/Slot weight view;
the public MLP implementation and native quantization methods are unchanged.

Both ordinary Slot and BF16 Home execution honor the configured SwiGLUStep
limit. A missing or zero limit retains the MoonEP default of 7.0. Quantized Home
supports W4A8MXFP with SiLU only and rejects other activations before device work.
The old quantized Home path accepted some non-SiLU configurations while executing
SiLU; that unchecked behavior is not restored.

MXFP8 selection recognizes both the torch-npu E8M0 format identifier emitted by
v0.30 quantization methods and the native torch E8M0 tensor dtype. An unspecified
scale format retains the implicit-E8M0 convention. A byte-carrier dtype alone
does not declare E8M0 semantics. The selector still checks original contiguous
BF16 input, E4M3 activation metadata, E8M0 runtime support and SDK wire geometry;
unsupported formats, missing ABI symbols and execution errors do not retry BF16.

Stage events are recorded only when the routed layer returns shared-expert
milestones and shared-expert multistream overlap is enabled. The GMM2 event is
recorded after activation/quantization and before GMM2. Dispatch and combine
events keep their existing stage boundaries. This optimization does not remove
prefetch-ready waits, SHMEM barriers or dedicated UDMA QP dependencies.

## Configuration

Enable the integration with:

```text
--enable-expert-parallel --enforce-eager
--additional-config '{"enable_moonep": true}'
```

The engine switch defaults to false. Existing environment parameter names and
defaults are retained:

| Parameter | Default / meaning |
| --- | --- |
| MOONEP_TRANSPORT_TYPE | mte; mte or udma |
| MOONEP_SHMEM_BOOTSTRAP_TYPE | ipport; ipport or uid |
| MOONEP_SHMEM_BOOTSTRAP_PORT | 28765 |
| MOONEP_KEEP_MC2 | 0; retain eligible native MC2 steps when enabled |
| MOONEP_REDUNDANT_EXPERT_SLOT_COUNT | unset; resolved from local expert count |
| MOONEP_CACHED_REDUNDANT_EXPERT_COUNT | 0; persistent remote-expert cache disabled |
| MOONEP_PREFETCH_SIDE_STREAM_OVERLAP | dispatch; none, dispatch, dispatch\|home_gemm1 |
| MOONEP_POOL_TOKEN_CAPACITY_TIERS | unset; automatic capacity ladder |
| MOONEP_POOL_MAX_TIER_COUNT | 32 |
| MOONEP_POOL_MEMORY_BUDGET_MB | 2048 MiB per-shape estimate, not a global VRAM limit |
| MOONEP_MIN_DISPATCH_TOKENS | 128; token-count floor for capacity selection |

SDK-owned planning/deduplication and barrier settings remain documented in
the companion SDK. Collective settings must agree across EP ranks.
UDMA overlap retains the dedicated-QP requirement. The existing SDK runtime
rejects expandable NPU allocator segments; keep
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` for MoonEP.

## SHMEM compatibility with memfabric

SHMEM's device-state getter must have local symbol binding when memfabric is
loaded in the same worker. Both libraries otherwise export the legacy
`HybmGetInitDeviceId()` name, allowing SHMEM to read memfabric's uninitialized
device number and fail initialization. The SDK applies the SHMEM internal visibility patch at build time. It also applies
a device-ID normalization patch for SHMEM user buffers. Build the SDK with both
patches and keep memfabric normally importable.

## Previous v0.30.0 validation environment

Validation uses host 118, container `xzx_moonep_latest`, with Python 3.12.13,
torch 2.10.0 and torch-npu 2.10.0.post7. Fresh-process checks resolve vLLM,
vllm-ascend and ascend-moonep to the selected latest workspace. The Python 3.12
aarch64 SDK extension and unified MXFP8/publish library ABI load successfully.
These checks do not initialize distributed communication or launch NPU kernels.

Verify package origins, the native extension selected for the interpreter
and architecture, loaded SDK/SHMEM/device libraries, and their hashes in a
fresh process before any device validation. Build compatible candidates
inside the new workspace if necessary; preserve the old installed artifacts.

## Validation handoff

For changes to dispatch or combine, validate BF16 and MXFP8 on multiple ranks
with empty and uneven batches, changing capacities, and optional shared experts.
For changes to weight management, also check the prefetch and cache event order.
Use matched Off/On configurations for performance comparisons and record the
exact revisions and settings with the results.
