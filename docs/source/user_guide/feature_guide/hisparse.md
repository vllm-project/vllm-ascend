# HiSparse on Ascend

HiSparse keeps active sparse-attention KV rows in NPU Resident/Hot storage and
refills selected missing rows from Host memory. Core owns HMA allocation,
Coordinator admission/eviction, request lifetimes, prefetch and mirror completion.
The Ascend backend supplies residency resolution, transfers and the existing SFA
Attention interface. Attention core arithmetic is unchanged.

## Supported configurations

- Experimental sparse RoPE MLA/SFA with Model Runner V2.
- BF16 KV and native Ascend 950 C8: E4M3FN main/Indexer values with FP32 scales.
  Main and Indexer cache dtypes are independently selectable.
- Private Host pools per tensor-parallel rank and Core eager Host mirroring.
- Ascend 950 with a SIMT-capable CANN 9.2 toolchain. A2/A3 are not newly qualified.
- TopK must satisfy the existing SFA requirement of divisibility by 128.

C4, TurboQuant, INT8-format C8, context parallelism and external P/D transport
are outside this backend's current support.

## Build the resolver

Use the paired merged vLLM version in `.github/vllm-main-verified.commit` and
install vllm-ascend normally in the matching Torch/torch-npu/CANN environment.
The standard CMake build compiles `csrc/hisparse/hisparse_resolver.cpp` using
CANN's ASC integration when the Ascend 950 SIMT headers are available. It links
the kernel into `vllm_ascend_C` and registers
`torch.ops._C_ascend.hisparse_resolve_residency` with the other Ascend operators;
there is no separate resolver build or library-loading step.

The architecture and usable per-AIV UB budget are selected for the build target
in `csrc/hisparse/CMakeLists.txt`; these are hardware/compiler parameters.
The kernel sizes its hash table from the request union and available UB. An
unsupported capacity or missing extension raises an error before execution;
there is no Triton resolver fallback. Builds for other targets retain their
existing operators without registering this resolver.

## Enable HiSparse

Set `VLLM_USE_V2_MODEL_RUNNER=1` and add these options to the compatible model's
normal serving command:

```bash
--block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

For native A5 C8, replace the two dtype options with:

```bash
--kv-cache-dtype fp8 \
--attention-config '{"indexer_kv_dtype":"fp8","hisparse_config":{"eager_host_mirror":true}}'
```

| Parameter | Meaning |
| --- | --- |
| `hisparse_config` | Enables Core HiSparse for the supported sparse Attention backend. |
| `eager_host_mirror` | Keep `true` for this foundation; Core maintains the Host KV mirror. |
| `host_pool_gib` | Host pool size per rank. Size it for the workload; the example reserves 4 GiB per rank. |
| `device_buffer_size` | Per-request Hot capacity from Core's HiSparse configuration. It must cover the selected verification-row union. |
| `kv_cache_dtype` / `indexer_kv_dtype` | Select main and Indexer KV formats independently from model-weight quantization. |

## Capacity and correctness notes

The resolver preserves the complete MTP query-row union before choosing LRU
victims, and produces the refill plan on device. Hot capacity must fit both the
int16 LRU representation (at most 32768 rows) and the compiled target's UB budget.
These checks do not change the configured TopK or discard selected tokens.

Native C8 main rows contain 512 E4M3FN bytes, 64 BF16 RoPE values and four FP32
scales (656 bytes total). Indexer rows contain 128 E4M3FN keys and one FP32 scale.
Views preserve the shared-slab page strides and layer offsets. The existing
quantizer and Attention operations are reused; payloads are copied as bytes.
With MXFP8 projection weights, C8 reuses SFA's existing Prolog V3 preprocessing
and publishes its staged rows through the HiSparse Resident mapping. Other
supported projection schemes retain the native preprocessing path.

C8 Decode consumes the resolver's mapped, stably compacted indices directly.
The refill plan retains separate virtual page-row IDs. Prefill keeps its existing
layout bridge, including stable ordering and duplicate selections.
Decode's padded query-shape lengths and physical KV extent are prepared before
capture and owned by the current binding. Per-round request mappings and valid
rows remain current. Native C8 Attention already clears empty query outputs;
the adapter returns that output without another index scan and output mask.

Request-row mapping reuses Core's device implementation, with separate storage
for each captured MTP graph owner and Ascend's padded input extent. Indexer
writes reuse the existing SFA store with independent page strides for keys and
scales, preserving the original payload bits. Small Host-mirror submissions use
PTA's ordered queue with immutable scalar descriptors; full-layer batches keep
the batch-copy API and Core's completion and payload-ownership boundaries.

Keep completion-before-unregister and graph allocation ownership intact.
Overwriting KV still invalidates any old Hot copy. Equal batch shapes do not
permit reusing stale request mappings or lengths. No upstream vLLM source change
or unmerged Core PR is required.

## Validation and performance

Ascend 950 eager/Graph component and TP8/MTP3 full-model checks cover the
supported foundation. Performance depends on the workload, residency and model
configuration. See [PR #20926](https://github.com/vllm-project/vllm-ascend/pull/20926)
for exact source pins, commands, measured ON/OFF results and remaining
qualification limits.

Formal performance runs exclude profiling. Reported miss and H2D counters are
asynchronous software observations, not physical DMA measurements. Qualification
on one workload does not establish dataset accuracy or support for other devices.
