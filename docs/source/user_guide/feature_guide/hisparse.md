# HiSparse on Ascend

## Feature Introduction

HiSparse keeps a Host mirror of sparse MLA KV rows and refills only the missing
rows selected by attention. This reduces the NPU cache capacity needed for long
contexts. The experimental Ascend backend requires a SIMT-capable CANN 9.2
toolchain on Ascend 950.

### Working Principle

HiSparse keeps recent KV rows in the NPU Resident cache and uses the NPU Hot
cache for rows refilled from Host memory. The resolver checks the selected rows
against both NPU caches. Resident and Hot
hits can be read directly. For misses, HiSparse refills only the missing rows
from the Host KV mirror into Hot before attention runs. With speculative decoding,
the resolver considers the selected rows for all query tokens of a request
together before evicting Hot rows.

### Usage Scenarios

Use HiSparse with Model Runner V2 for sparse RoPE MLA/SFA models serving long
contexts. BF16 KV and native Ascend 950 C8 (E4M3FN values with FP32 scales) are
supported, with independent Main and Indexer KV dtype selection.

### Constraints and Limitations

| Category | Constraint |
| --- | --- |
| Hardware and software | Ascend 950 with a SIMT-capable CANN 9.2 toolchain. |
| Cache formats | C4, TurboQuant and INT8-format C8 are outside the current support scope. |
| Parallelism and transport | PCP=2 is validated for native C8 on one Ascend 950 node. DCP and external P/D transport remain outside the current support scope. |
| Host storage | Use private Host pools per tensor-parallel rank and Core eager Host mirroring. |
| TopK | The existing SFA requirement of divisibility by 128 applies. |
| Feature mutual exclusion | HiSparse and `additional_config.sparse_kv_offload_config.enabled` cannot be enabled together. |

## Feature Usage

### Usage Example

Set `MODEL_PATH` to a compatible sparse MLA model and start the server with
Model Runner V2:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$MODEL_PATH" \
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

### Prefill Context Parallelism

For native C8 on an eight-device Ascend 950 node, add the following to the
Model Runner V2 C8 command above:

```bash
--tensor-parallel-size 4 \
--prefill-context-parallel-size 2 \
--enable-expert-parallel \
--attention-config '{"indexer_kv_dtype":"fp8","hisparse_config":{"eager_host_mirror":true,"device_buffer_size":10240}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":64}}'
```

Use the direct `HiSparseConnector` with private per-rank Host pools. This path
installs an Ascend query/write-layout adapter; no Core source patch is required. PCP gathers packed C8
KV rows before Resident writes and Host mirroring; sparse attention still uses
local query rows. No additional custom kernel is needed.

GLM-5.3 C8 has been validated with EP, full decode graphs and MTP3. The Hot
buffer must hold the union of selected rows across verification tokens; 10240
rows was used for top-k 2048 and three speculative tokens. Keep the same
parallelism, graph and speculative settings when comparing offload on/off.
Check both generated outputs and increasing miss counters under a constrained
Resident budget; an all-Resident run does not exercise Host refill.

### Verifying the Feature

Send a request and check the server's metrics endpoint for the HiSparse miss
counter. The counter can stay at zero when selected rows are already on the NPU.

```bash
curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:hisparse_cache_misses_total'
```

## Configuration Parameters

| Parameter | Meaning |
| --- | --- |
| `hisparse_config` | Enables Core HiSparse for the supported sparse Attention backend. |
| `eager_host_mirror` | Keep `true` for this foundation; Core maintains the Host KV mirror. |
| `host_pool_gib` | Host pool size in GiB per tensor-parallel rank. Size it for the planned context lengths and concurrency; the example reserves 4 GiB per rank. |
| `device_buffer_size` | Per-request Hot capacity in KV rows. It must hold the deduplicated union of rows selected during speculative verification and fit the resolver's on-chip UB budget. |
| `kv_cache_dtype` / `indexer_kv_dtype` | Select main and Indexer KV formats independently from model-weight quantization. |
