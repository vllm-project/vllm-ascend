# MoonEP on vLLM Ascend v0.30.0

## Source baselines and migration status

This branch migrates the final MoonEP implementation from `moonep_v0.23.0`
(`73d2f9e4e21e911ec86682684bc46d862674a595`) onto
`releases/v0.30.0` (`56279e424b0636afe936d011ff65b9c4e6a6c82e`).

The companion vLLM checkout is `releases/v0.30.0`
(`ced6857afa0ea7b2e3f0846a62e1394e90f15607`). The SDK baseline is
`ascend-moonep` `urma_dev` (`6de30d77d0f21721bce35a73018160e56fb9d656`).
This migration changes vllm-ascend source only. It does not install packages,
rebind editable installations, rebuild native libraries, or update submodules.

The workspace for this work is
`/mnt/share/x00893896/vllm_ascend_moonep_integration_latest`.
The previous workspace and source branch remain separate baselines.

**Status: source adaptation with dependency-free contract checks. NPU operators,
multirank collectives, model correctness, and performance are not yet validated.**
Copied native binaries are not evidence of ABI compatibility.

## Execution integration

- `AscendRoutedExperts` registers the actual owner of expert Parameters.
  Runtime allocation and weight publication occur after model weight loading
  in both model runners, before the first forward.
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

## Independent validation environment

Prepare a dedicated environment later and install all three packages from
the selected workspace. The v0.30 branch pins torch 2.10.0 and
torch-npu 2.10.0.post7. The current container installation still points to
the previous source workspace and is not a v0.30 validation environment.

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

The 26 retained MoonEP regression files require the dedicated CPU/framework
test environment. Their complete runtime execution remains pending.
Then validate native imports and library provenance, BF16 and MXFP8 operators,
multiple ranks with empty/uneven batches and changing capacities, prefetch/cache
event ordering, shared experts, both model runners, model correctness, and
matched Off/On performance. Preserve raw evidence and report unsupported
combinations instead of silently changing the requested configuration.
