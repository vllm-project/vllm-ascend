# SFA PD RD2H KV Transfer

## Overview

`SFAPDRD2HConnector` transfers SFA KV cache from a dedicated Prefill node to a
Decode node in a Prefill-Decode disaggregated deployment. RD2H means remote
device to local host: Prefill exposes its layer-wise NPU KV cache, and Decode
pulls it through MemFabric into the storage managed by Sparse KV Cache Offload.

On Decode, the main KV cache is stored in the host offload pool, while the
indexer KV cache remains in rank-local NPU memory. This reduces Decode-side NPU
memory usage without loading the full KV cache back to the NPU for every decode
step.

The connector provides the transfer path only. It does not enable Prefill or
Decode offload by itself.

## Feature Relationship

Sparse KV Cache Offload is required on Decode. Layerwise Prefill KV Cache
Offload is optional on Prefill:

- [Sparse KV Cache Offload](sparse_kv_cache_offload.md) owns the Decode-side
  destination layout and offloads the full KV cache to host memory. Follow this
  guide for requirements, installation, and Decode configuration. Do not
  enable Sparse KV Cache Offload on Prefill.
- [Layerwise Prefill KV Cache Offload](layerwise_prefill_kv_offload.md) reduces
  Prefill-side NPU memory usage by reusing layer-wise KV buffers. Without this
  optional feature, configure `SFAPDRD2HConnector` directly on Prefill. When it
  is enabled, compose `AscendStoreConnector` and `SFAPDRD2HConnector` through
  `MultiConnector` on Prefill.

## Deployment Notes

- Set `transfer_backend` to `memfabric` on both Prefill and Decode.
- Prefill TP must be greater than or equal to, and divisible by, Decode TP.
- Use the layerwise proxy described in the
  [Layerwise KV Pool guide](layerwise_kv_pool.md) for request routing and
  metaserver rendezvous.
- Only Decode binds connector control sockets. Reserve
  `decode_data_parallel_size * decode_tensor_parallel_size` consecutive ports
  starting at `kv_port`; Prefill may use the same base value for consistency.

For the connector architecture, control protocol, tensor-parallel ownership,
and completion semantics, see the
[SFA PD RD2H Connector Design](../../developer_guide/Design_Documents/sfa_pd_rd2h_connector.md).
