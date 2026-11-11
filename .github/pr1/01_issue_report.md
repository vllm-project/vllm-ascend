# [Bugfix] Native OffloadingConnector crashes on hybrid attention models (GDN + full attention) due to non-coalescible interleaved layout

## Describe the bug

vLLM-Ascend's native KV offloading connector (`OffloadingConnector`) crashes with a `ValueError` during worker initialization when serving hybrid attention models on Ascend NPUs. Hybrid attention models (e.g. Qwen3.5, model_type=`qwen3_5`) combine GDN linear attention layers with full attention layers, producing two KV cache groups with different physical layouts. On Ascend NPUs, the model runner packs both groups' storage into a unified pool where Mamba state slots and full-attention KV blocks are interleaved, resulting in a non-coalescible layout (`stride_bytes` ≠ `physical_block_bytes`).

The `AscendOffloadingConnectorWorker.register_kv_caches()` method attempts to build int8 block views for ALL KV cache groups, including MambaSpec groups whose tensors fail the coalescing check, crashing all TP workers and preventing the service from starting.

## To Reproduce

```bash
vllm serve /path/to/Qwen3.5-27B \
    --served-model-name qwen3.5-27B \
    --tensor-parallel-size 4 --max-model-len 131072 \
    --gpu-memory-utilization 0.85 --dtype bfloat16 \
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":137438953472,"blocks_per_chunk":8,"spec_name":"CPUOffloadingSpec"}}' \
    --host 0.0.0.0 --port 8002
```

Model config confirms the hybrid structure:
```json
{
    "model_type": "qwen3_5",
    "num_hidden_layers": 64,
    "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention", "... (3:1 repeating, 48 GDN + 16 full)"],
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_conv_kernel_dim": 4
}
```

## Expected behavior

The offloading connector should skip non-offloadable groups (MambaSpec) and only manage full-attention groups' KV blocks. The Mamba/GDN states remain in NPU and are managed by vLLM's `HybridKVCacheCoordinator` state cache mechanism. The upstream `OffloadingConnector._bounding_group_ids` already provides the scheduler-side semantics for bounding load depth by non-offloaded groups.

This is the same design pattern used by MooncakeConnectorV1 (vllm-ascend's PD connector) and AscendStoreConnector, which both handle hybrid models by processing full-attention groups normally and excluding MambaSpec groups from block-level offload.

## Actual behavior

All 4 TP workers crash during `register_kv_caches`:

```
ValueError: Cannot coalesce a non-contiguous Ascend KV cache layout: stride_bytes=131072, physical_block_bytes=65536
```

Traceback:
```
File "vllm_ascend/worker/model_runner_v1.py", line 4692, in initialize_kv_cache
    get_kv_transfer_group().register_kv_caches(kv_caches)
File "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py", line 109, in register_kv_caches
File "vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/offloading_connector.py", line 140, in <genexpr>
    _make_int8_block_view(
File "vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/offloading_connector.py", line 73, in _make_int8_block_view
    raise ValueError(
ValueError: Cannot coalesce a non-contiguous Ascend KV cache layout: stride_bytes=131072, physical_block_bytes=65536
```

Root cause: in the hybrid model's NPU KV cache pool, Mamba state slots and full-attention KV blocks are interleaved (each 64KB Mamba state slot is followed by a 64KB full-attention KV block), producing a stride of 128KB with a payload of 64KB. The `_make_int8_block_view()` function requires stride == payload for coalescing multiple physical blocks into a manager block, and correctly refuses the interleaved layout.

## Environment

- vllm-ascend: v0.30.0rc2 (quay.io/ascend/vllm-ascend:v0.30.0rc2)
- Hardware: Ascend 910B3 ×4 (TP4)
- Model: Qwen3.5-27B (model_type=qwen3_5, 48 GDN linear + 16 full attention)
- CANN: 9.2.0

## Proposed fix

Skip MambaSpec groups in `AscendOffloadingConnectorWorker.register_kv_caches()`. The mamba states remain in NPU and are managed by vLLM's `HybridKVCacheCoordinator`. Only full-attention groups participate in native offloading. This matches the design pattern already used by MooncakeConnectorV1 and AscendStoreConnector for hybrid models.

A PR implementing this fix will be submitted referencing this issue.
