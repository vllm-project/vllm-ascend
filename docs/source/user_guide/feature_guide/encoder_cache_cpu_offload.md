# Encoder Cache CPU Offload

`ECCPUConnector` stores multimodal encoder outputs in a shared CPU mmap. A
repeated image, audio item, or other multimodal input can be restored from CPU
memory instead of running the encoder again.

## Support scope

- vLLM V1 and V2 model runners
- An Ascend device with a CANN build that provides both `aclrtMemcpyBatchAsync` and
  `aclrtHostRegisterV2(..., ACL_HOST_REG_PINNED)`
- Linux kernel newer than 5.10, as required by CANN when converting an mmap
  allocation into page-locked Host memory
- A single host using `/dev/shm`
- `ec_both` role

Cross-node encoder-cache transfer and NIXL/P2P transfer are not supported by
this connector.

The Ascend worker registers its shared mmap as real page-locked Host memory.
If pinned mmap registration or batch memcpy is unavailable, initialization
fails with an explicit error. It does not silently use pageable memory or the
per-copy `aclrtMemcpyAsync` fallback.

## Configuration

The following example allocates 1 GiB for encoder-cache entries:

```bash
vllm serve <model-path> \
  --ec-transfer-config '{
    "ec_connector": "ECCPUConnector",
    "ec_role": "ec_both",
    "ec_connector_extra_config": {
      "ec_cpu_bytes": 1073741824
    }
  }'
```

`ec_cpu_bytes` must be a positive integer large enough for at least one
encoder-cache block. The corresponding mmap is created under `/dev/shm`, so
the host must also have enough tmpfs capacity. Tensor-parallel workers map the
same file; each worker process registers its own virtual mapping with CANN.

## Transfer behavior

On a cache miss, the encoder output is copied from NPU to the pinned mmap with
one batched D2H operation. On a hit, the blocks are copied back with one
batched H2D operation. Only TP rank 0 and PCP rank 0 write duplicate encoder
outputs, while every consumer worker can load its local NPU tensor.

The encoder output and D2H save are submitted to the current compute stream,
so normal stream ordering guarantees that the encoder output is ready before
the copy starts. Loads use a dedicated stream, and the current compute stream
waits for that load stream before the restored tensor is consumed.

During shutdown, the NPU device is synchronized before the Host mmap is
unregistered from CANN, closed, and removed. If device synchronization fails,
the mapping is deliberately left intact for process-exit cleanup rather than
being unmapped while an asynchronous DMA may still reference it.
