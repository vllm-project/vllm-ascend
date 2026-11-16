# MoonEP on vLLM Ascend main

## Current main2main baseline

The integration branch `moonep_main` replays the six MoonEP commits through
`moonep_v0.30.0` (`8d173bb61238`) on vllm-ascend main (`4e11513255d8`).
Its verified vLLM revision is `ba15685ab7f4` (main2main 1106).
The working directory is `/mnt/share/x00893896/vllm_ascend_moonep_integration_main`.
Model Runner V2 initializes MoonEP after weight loading inside its resource
cleanup scope. Device and performance validation on this pair is pending.

## Previous v0.30.0 migration and validation

The v0.30.0 branch migrated the final MoonEP implementation from `moonep_v0.23.0`
(`73d2f9e4e21e911ec86682684bc46d862674a595`) onto
`releases/v0.30.0` (`56279e424b0636afe936d011ff65b9c4e6a6c82e`).

The companion vLLM checkout is `releases/v0.30.0`
(`ced6857afa0ea7b2e3f0846a62e1394e90f15607`). The SDK feature baseline is
`ascend-moonep` `urma_dev` (`6de30d77d0f21721bce35a73018160e56fb9d656`).
The validation container uses `urma_dev_vllm_ascend_v0.30.0` with the CANN 9.2
header-layout compatibility commit `5a312f186bc6bf197df223f67d2cd4f7c9b1274d`.
This migration changes vllm-ascend source only. It does not install packages,
rebind editable installations, rebuild native libraries, or update submodules.

The workspace used for that migration was
`/mnt/share/x00893896/vllm_ascend_moonep_integration_latest`.
The previous workspace and source branch remain separate baselines.

**Status: MoonEP CPU regression and fresh-process native-loader checks pass.
NPU operators, multirank collectives, model correctness, and performance remain
unvalidated.** Successful loading does not establish device numerical correctness.

## Execution integration

- `AscendRoutedExperts` registers the actual owner of expert Parameters.
  Runtime allocation and weight publication occur after model weight loading
  in Model Runner V2, before the first forward.
- SHMEM communication methods use the v0.30 per-configuration registry.
  Native hardware-policy selection remains the baseline when MoonEP is off.
- Dispatch uses real token counts, rank-symmetric capacity selection, and the
  existing SDK planning/dispatch/prefetch/weighted-combine protocol.
- `SlotMLPLayerView` supplies Home/Slot aliases to native quant-method hooks.
  Replacing only `MoEMlpComputeInput.weights` is insufficient in v0.30 because
  the hooks obtain tensors from `layer`. The view supplies weights and scales
  without swapping live model Parameters on each forward.
- BF16 dispatch, automatic W4A8MXFP/MXFP8 dispatch, effective-row unpacking,
  fused publication, FP32 weighted combine, buffer reuse, capacity pools,
  optional expert caching, and dispatch/Home-GEMM1 overlap retain their SDK
  implementation.
- Prefetch identifies the weight owner at the actual MoE call. Shared experts
  receive real dispatch, GMM2, and combine milestones; these execution events
  are not the removed development diagnostics.

MoonEP requires expert parallelism over multiple ranks, contiguous linear
expert placement, eager execution, and supported unsplit expert tensors.
Dynamic EPLB, mix placement, expert bias, split expert lists, and MoE LoRA are
rejected. Home GEMM supports BF16 gated activations from the existing path and
W4A8MXFP with SiLU; other Home backends fail explicitly.
Native Off retains its own EPLB and communication paths.

The development timers, load-accounting diagnostics, simulated routing,
predicted planning, and plan dumps removed from the source branch are not
reintroduced. Historical experiments remain on the old branch.

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
| MOONEP_MIN_DISPATCH_TOKENS | 128; existing minimum selected capacity |

SDK-owned planning/deduplication and barrier settings remain documented in
the companion SDK. Collective settings must agree across EP ranks.
UDMA overlap retains the dedicated-QP requirement. The existing SDK runtime
rejects expandable NPU allocator segments; keep
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` for MoonEP.

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

The generic launchers require the intended interpreter and check source
origins before starting. They do not install anything, select physical cards,
or choose network interfaces. Load the CANN environment and configure the
authorized devices/network before use.

```bash
MODEL=/path/to/model \
VLLM_PYTHON=/path/to/independent/env/bin/python \
bash scripts/dsv4flash_service/service/start_service_mode.sh slot
```

Use `off` for the baseline. The DP4 wrapper uses DP_SIZE=4; the generic
launcher defaults to DP_SIZE=8 and TP_SIZE=1. Pass additional vLLM arguments
after the mode, including official profiler configuration when needed.
Use unique output locations for diagnostics.

## Validation handoff

Dependency-free source-contract checks can run without torch or NPU imports:

```bash
/path/to/python -I -B tests/ut/ops/test_moonep_v030_contract.py -v
bash -n scripts/dsv4flash_service/service/start_service_mode.sh
bash -n scripts/dsv4flash_service/service/start_service_mode_dp4.sh
git diff --check
```

Run the MoonEP CPU regressions with the validation interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/python3.12.13/bin/python -m unittest discover \
  -s tests/ut/ops -p 'test_moonep*.py' -v
```

These tests use CPU tensors and mocked device/framework boundaries; they cover
native quant metadata, activation limits, Home/Slot aliases, token/count bounds,
cache/prefetch ordering and conditional shared-expert events. They do not prove
NPU execution. Device validation must additionally cover BF16 and MXFP8 operators,
multiple ranks with empty/uneven batches and changing capacities, prefetch/cache
event ordering, shared experts, Model Runner V2, model correctness, and
matched Off/On performance. Preserve raw evidence and report unsupported
combinations instead of silently changing the requested configuration.
