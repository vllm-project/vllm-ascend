# HiSparse on Ascend

## Feature Introduction

HiSparse keeps a Host mirror of sparse MLA KV rows and refills only the missing
rows selected by attention. This reduces the NPU cache capacity needed for long
contexts. The experimental Ascend backend supports A2/A3 through the arch22
resolver and 950PR&950DT Products through the SIMT resolver. 950PR&950DT Products require a SIMT-capable CANN 9.2
toolchain; build the custom operators for the target architecture.

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
contexts. A2/A3 support INT8 C8 KV: Main KV stores INT8 values, BF16 RoPE and
FP32 scales; Indexer KV stores INT8 values with FP16 scales. 950PR&950DT Products retain BF16 KV
and native C8 (E4M3FN values with FP32 scales), with independent Main and
Indexer KV dtype selection. The A2/A3 extension does not add a BF16 HiSparse
serving path or change the existing non-HiSparse BF16 path.

### Constraints and Limitations

| Category | Constraint |
| --- | --- |
| Hardware and software | A2/A3 with the arch22 custom operators, or 950PR&950DT Products with a SIMT-capable CANN 9.2 toolchain. |
| Cache formats | A2/A3: INT8 C8. 950PR&950DT Products: BF16 or native E4M3FN C8. C4 and TurboQuant are outside the current support scope. |
| Parallelism and transport | Context parallelism and external P/D transport are outside the current support scope. |
| Host storage | Requires AIV Host-memory mapping support from the device and runtime/driver stack. Use private Host pools per tensor-parallel rank and Core eager Host mirroring. |
| TopK | The existing SFA requirement of divisibility by 128 applies. |
| Feature mutual exclusion | HiSparse and `additional_config.sparse_kv_offload_config.enabled` cannot be enabled together. |

## Feature Usage

### Usage Example

Set `MODEL_PATH` to a compatible sparse MLA model and start the server with
Model Runner V2. The following example uses BF16 KV on 950PR&950DT Products:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$MODEL_PATH" \
  --block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

For native C8 on 950PR&950DT Products, replace the two dtype options with:

```bash
--kv-cache-dtype fp8 \
--attention-config '{"indexer_kv_dtype":"fp8","hisparse_config":{"eager_host_mirror":true}}'
```

For A2/A3 INT8 C8, use a compatible INT8 KV model and replace the two dtype
options with:

```bash
--kv-cache-dtype int8 \
--attention-config '{"indexer_kv_dtype":"int8","hisparse_config":{"eager_host_mirror":true}}'
```

These options select KV-cache quantization independently of model-weight
quantization; keep the weight-quantization options required by the model.

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
