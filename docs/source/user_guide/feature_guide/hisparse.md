# HiSparse on Ascend

## Overview

Sparse attention selects a subset of KV rows for each query. HiSparse keeps
recent KV rows in the NPU Resident cache and uses the NPU Hot cache for rows
refilled from Host memory.

The resolver checks the selected rows against both NPU caches. Resident and Hot
hits can be read directly. For misses, HiSparse refills only the missing rows
from the Host KV mirror into Hot before attention runs. With speculative decoding,
the resolver considers the selected rows for all query tokens of a request
together before evicting Hot rows.

## Supported Scenarios

- Experimental sparse RoPE MLA/SFA with Model Runner V2.
- BF16 KV and native Ascend 950 C8: E4M3FN main/Indexer values with FP32 scales.
  Main and Indexer cache dtypes are independently selectable.
- Private Host pools per tensor-parallel rank and Core eager Host mirroring.
- Ascend 950 with a SIMT-capable CANN 9.2 toolchain. A2/A3 are not newly qualified.
- TopK must satisfy the existing SFA requirement of divisibility by 128.

C4, TurboQuant, INT8-format C8, context parallelism and external P/D transport
are outside this backend's current support.

## Usage

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
| `host_pool_gib` | Host pool size in GiB per tensor-parallel rank. Size it for the planned context lengths and concurrency; the example reserves 4 GiB per rank. |
| `device_buffer_size` | Per-request Hot capacity in KV rows. It must hold the deduplicated union of rows selected during speculative verification and fit the resolver's on-chip UB budget. |
| `kv_cache_dtype` / `indexer_kv_dtype` | Select main and Indexer KV formats independently from model-weight quantization. |
