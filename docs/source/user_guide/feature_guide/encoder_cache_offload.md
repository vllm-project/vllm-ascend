# Encoder Cache Offload

Encoder-cache offloading reuses multimodal encoder outputs when the same input
is requested again. Ascend supports two connectors:

- `ECCPUConnector`: cache encoder outputs in CPU memory.
- `ECTieringConnector`: extend the CPU cache with an optional filesystem tier.

Both connectors use the existing Ascend CPU offload worker for NPU transfers.
They cache encoder outputs, not the decoder KV cache.

## ECCPUConnector

`ECCPUConnector` keeps encoder outputs in a shared, pinned CPU mmap. It is
intended for reuse within a running engine on a single host.

### Requirements

- Use a paired vLLM version that provides `ECCPUConnector` and enable Model
  Runner V2 with `VLLM_USE_V2_MODEL_RUNNER=1`.
- Build vllm-ascend against a CANN stack providing both `aclrtMemcpyBatchAsync`
  and `aclrtHostRegisterV2(..., ACL_HOST_REG_PINNED)`. The runtime must support
  registering the shared mmap as pinned Host memory. Initialization fails if
  the required support or registration is unavailable.
- Provide enough `/dev/shm` capacity for the CPU cache and other engine shared
  memory allocations. In a container, configure its shared-memory limit
  accordingly.

### Configuration

This example enables local reuse with a 1 GiB CPU cache:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <model-path> \
  --ec-transfer-config '{
    "ec_connector": "ECCPUConnector",
    "ec_role": "ec_both",
    "ec_connector_extra_config": {
      "ec_cpu_bytes": 1073741824
    }
  }'
```

Set `ec_cpu_bytes` to the CPU cache budget in bytes. Choose a positive integer
large enough to hold the encoder outputs you want to reuse. Allocation is in
whole blocks, so the usable capacity is rounded down to a block boundary.
The mmap lives under `/dev/shm`; workers sharing it each register their own
virtual mapping with CANN.

Use `ec_both` for this local offload configuration. Ascend does not support
NIXL/P2P transfer through this connector. Omit `ec_enable_nixl` entirely:
including the key raises an error even when its value is `false`.

### Cache and transfer behavior

When an encoder output needs to be computed and CPU space is available, the
worker saves it to CPU memory. Later CPU hits restore the output to the NPU.
The CPU cache uses LRU eviction; an entry that has been evicted must be
computed again unless it is still available in the NPU encoder cache. CPU
entries are not retained for reuse after an engine restart.

D2H saves use a transfer stream that waits for the encoder's compute stream.
H2D loads use a transfer stream that the consuming compute stream waits for.
Transfer completion events protect buffers until DMA has finished. Only TP
rank 0 and PCP rank 0 save the replicated encoder output; consumer workers
load their own NPU tensors.

## ECTieringConnector

`ECTieringConnector` adds a filesystem cache behind the CPU cache. Ascend
reuses vLLM's tiering scheduler, cache management, and filesystem implementation,
and uses the same NPU transfer worker as `ECCPUConnector`.

### Requirements

The CPU offload requirements above also apply. In addition:

- The paired vLLM installation must provide and register `ECTieringConnector`.
  Ascend does not register this adapter when the upstream connector is absent.
- Configure a writable filesystem directory with enough free space for the
  selected cache budget. A filesystem hit must first be restored into the CPU
  cache before it can be loaded onto the NPU.

### Configuration

This example enables a 1 GiB CPU cache and a 16 GiB filesystem tier:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <model-path> \
  --ec-transfer-config '{
    "ec_connector": "ECTieringConnector",
    "ec_role": "ec_both",
    "ec_connector_extra_config": {
      "ec_cpu_bytes": 1073741824,
      "secondary_tiers": [{
        "type": "fs",
        "root_dir": "/var/tmp/vllm-encoder-cache",
        "capacity_bytes": 17179869184
      }]
    }
  }'
```

`ec_cpu_bytes` has the same meaning as for `ECCPUConnector`. Use `ec_both` and
omit `ec_enable_nixl` here as well. `secondary_tiers` accepts at most one tier,
whose type must be `fs`. Omitting it or providing an empty list leaves only
the CPU cache enabled.

The filesystem tier accepts these settings:

| Setting | Meaning | Default |
| --- | --- | --- |
| `type` | Tier type; must be `"fs"`. | Required |
| `root_dir` | Absolute path to the cache directory. | Required |
| `capacity_bytes` | Positive cache capacity in bytes. | Required |
| `n_read_threads` | Positive number of filesystem read threads. | `4` |
| `n_write_threads` | Positive number of filesystem write threads. | `2` |
| `fsync` | Synchronize each temporary cache file before publishing it. | `false` |
| `backpressure.enabled` | Enable skipping new filesystem writes under write pressure. | `true` |
| `backpressure.high_water_s` | Stop accepting new cache writes when smoothed write time exceeds this many seconds per MiB. | `0.010` (10 ms/MiB) |
| `backpressure.low_water_s` | Resume writes below this smoothed time per MiB; must be positive and no greater than `high_water_s`. | `0.002` (2 ms/MiB) |

Cache files store complete allocated CPU blocks, including padding; capacity
accounting uses that full size.

### Cache behavior and persistence

After a new encoder output has finished saving to CPU, the connector submits
an asynchronous filesystem write. It writes through from CPU; it does not wait
for CPU eviction before creating the filesystem copy.

Requests can start asynchronous filesystem prefetch when they enter the queue.
A CPU miss that hits the filesystem restores the entry to CPU, then loads it
onto the NPU. While an asynchronous lookup or submitted read is pending, the
request can be deferred without blocking the scheduler on file I/O. If the
entry is absent, or a promotion cannot proceed, normal cache-miss handling
allows the encoder to compute it again.

Filesystem entries can survive an engine restart when the same cache directory
and compatible cache identity are used. Identity includes the model,
multimodal, and kernel configuration; changes can cause existing files to be
ignored. Files are a capacity-limited cache, not a durable export of encoder
outputs. Write backpressure, I/O failures, and eviction can prevent later reuse;
`fsync=false` also does not guarantee persistence across a system crash.
