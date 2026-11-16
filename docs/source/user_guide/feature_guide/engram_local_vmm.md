# Experimental local shared Engram VMM

On supported A3 systems, shared, CPU-offloaded INT8 DeepSeek-V4.1 Engram
tables automatically use node-local Host VMM. This backend maps host memory for each device; it avoids
registering the complete shared table through `aclrtHostRegisterV2`.

Use a real INT8 checkpoint and shared CPU offload:

```bash
export VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=60
vllm serve MODEL_PATH \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --shutdown-timeout 30
```

This version has not been validated with Mooncake P/D. Existing `shared_engram`
configurations use their own backend. Host VMM requires a build with the Host VMM library,
a compatible CANN/driver stack, and a shared `/dev/shm` namespace for local
ranks. Each sharing group has one checkpoint writer and separate mappings for
its codes and scales. This is not cross-node table sharing.
If the library cannot be loaded on every rank, the group uses the original
HostRegister backend. If VMM initialization fails after selection, startup
fails after coordinated cleanup; it does not automatically switch backends.

The timeout settings above were used for the evaluated multi-process shutdown;
they do not guarantee safe recovery from a lost peer or an unfinished remote
transfer. Stop admission and drain requests before shutdown. If a transfer
times out or a peer fails, process exit and the lifetime of memory still in
use by another device are not yet proven safe. Do not use this experimental
backend where that failure behavior is unacceptable.

If initialization or rollback fails, affected local VMM objects may be retained
in the worker process for diagnosis and coordinated cleanup. This retention is
not durable across worker exit, and service startup failure can terminate the
worker. The backend does not automatically recover failed initialization.
Do not attempt collective cleanup unless every member of the original sharing
group is alive and a recovery coordinator has established a consistent retry
order. A missing peer or a failed transfer requires an independently verified
device/transport isolation guarantee before resources can be reused.
