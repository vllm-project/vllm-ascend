# sharded_rdt on Ascend with HIXL

The public backend remains `sharded_rdt`. The Ascend plugin selects the HIXL
implementation on NPU; other platforms retain the upstream implementation.
This is an initial implementation requiring NPU and model validation before use
in a training service. CPU contract tests do not establish device correctness.

## Source baseline and components

This implementation synchronizes the planner, ownership rules, gather credits,
shared serve rings and layerwise replay from vLLM commit
`24420b7a341af37a9bc3186e6fc6062512e0aa58`. It is paired with vLLM Ascend
`934236001e6a902d9c23606e56d3ff258f73377c`.

- `vllm_ascend/distributed/weight_transfer/__init__.py` selects the worker and
  trainer classes under the existing registry key.
- `sharded_rdt_trainer.py` gathers checkpoint-format tensors, publishes local
  NPU IPC handles to a co-located Ray producer, and serves packed slices.
- `sharded_rdt_engine.py` bakes the model's loading plan and pulls the slices
  needed by each inference worker.
- `hixl_session.py` owns the CANN engine, aligned registrations, READ requests,
  polling and teardown.

The upstream factory currently rejects duplicate names and has no public
platform override API. The plugin wraps only the two existing `sharded_rdt`
lazy loaders, preserving their original loaders for other platforms. It never
registers `sharded_hixl`. This adapter can be removed when upstream supports
platform-specific registration. The two engines inherit the upstream planner,
ownership and pipeline rules, overriding the device and transport methods.
Those overrides track the pinned source; review them explicitly on upgrade.

## Interface flow

```text
WeightTransferTrainerFactory.trainer_init(ShardedRDTTrainerInitInfo, source)
  -> one producer actor on each trainer's node and physical NPU
  -> metadata / ownership / producer names sent to inference workers
  -> worker init: bake load plan, reserve and register fixed rings

each weight update:
  trainer source.iter_full() -> layerwise gather -> NPU stream synchronization
  -> reduce_tensor arguments -> producer.publish_group (NPU IPC rebuild)
  -> worker router.pull -> producer.rdt_produce_weights_batched
  -> producer applies the recorded view operations and packs 16-byte slices
  -> synchronize pack -> return (HIXL endpoint, address, byte count) over Ray
  -> worker transfer_async(endpoint, READ, local/remote descriptors)
  -> get_transfer_status until COMPLETED
  -> NPU scatter -> quantization / kernel copy -> finish_weight_update
```

Ray carries control messages and addresses, not tensor payloads. NPU IPC is
between the trainer and its local producer. HIXL moves the packed bytes between
producer and inference worker. The producer and trainer must share the same
physical NPU and compatible device visibility mappings.

The consumer's generation counters and NPU events prevent a receive ring slot
from being overwritten while scatter still reads it. The producer's sequence
number and shared-plan digest preserve the upstream serve-slot reuse protocol.
HIXL registration has 2 MiB aligned address and length; individual packed slices
retain the upstream 16-byte layout. Ring allocation retains sufficient backing
storage for the entire aligned region. Buffers never grow during replay.

## Prerequisites

Install the paired vLLM and this Ascend checkout, Ray, `torch_npu`, and a CANN
toolkit with usable HIXL support. Source the installed toolkit's environment in
both trainer and inference processes. Hardware, firmware, driver and CANN must
support the selected HIXL transport; verify this on the actual deployment.
Existing EPLB HIXL support alone is not proof of weight-transfer correctness.

The binding first imports the official `hixl` Python package. If absent, it uses
the existing `vllm_ascend.distributed.eplb.hixl_compat` binding. That fallback
depends on the local CANN C++ ABI; do not assume it supports arbitrary toolkit
versions. No NIXL package or Ray tensor-transport opt-in is used in the NPU path.

Verify network reachability between HIXL endpoints. Endpoint host selection
uses vLLM's `get_ip()` and its existing `VLLM_HOST_IP` configuration. Ports are
selected dynamically. Where required by your HIXL installation, have the
administrator configure `/etc/hixlep`; this implementation does not change it.

## Configuration

Inference retains the same backend name:

```bash
vllm serve /path/to/model \
  --distributed-executor-backend ray \
  --weight-transfer-config '{"backend":"sharded_rdt"}'
```

Configure TP/DP/EP for the actual model. EPLB must be disabled because dynamic
expert rearrangement invalidates the baked destination plan. The model loader
must support the upstream fake-tensor bake operations; unsupported data-dependent
loading operations fail at initialization.

On every trainer rank, initialize Ray in the same cluster/namespace as the
inference workers and load plugins before creating the trainer engine:

```python
from vllm.plugins import load_general_plugins
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm_ascend.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
)

load_general_plugins()
engine = WeightTransferTrainerFactory.trainer_init(
    ShardedRDTTrainerInitInfo(
        rank=trainer_rank,
        num_consumers=inference_worker_count,
        trainer_actor_namespace=ray_namespace,
        num_rdt_buffers=2,
    ),
    client=HTTPVLLMWeightSyncClient(inference_url),
    source=checkpoint_weight_source,
)
```

`checkpoint_weight_source` must implement the upstream `WeightSource` contract:
checkpoint parameter names, dtype/shape metadata and `iter_full()`, with names
contiguous by layer group. Distributed ranks must agree on metadata and ownership.
Use `ModuleSource` for models already exposing the checkpoint layout; fused
expert names may require a model-specific mapping. This snippet assumes the
trainer's distributed process group and NPU device are already initialized.

Keep the existing client's pause/update/resume bracket and call the trainer's
`send_weights` inside the update callback. See the paired upstream
`examples/rl/rlhf_sharded_rdt_small_ep.py` for the control-plane sequence; its
CUDA/NCCL resource setup must not be used unchanged on NPU.

## Validation

Run CPU contract tests from the Ascend repository root:

```bash
python -m pytest --confcutdir=tests/ut/distributed/weight_transfer \
  tests/ut/distributed/weight_transfer/test_sharded_rdt_hixl.py -q
```

On an idle Ray cluster with two NPU resources, run the direct transport check:

```bash
python examples/rl/sharded_rdt_hixl_smoke.py
```

It publishes two different byte patterns through the same registered source
buffer and checks the entire receive buffer after each READ. It does not test
FSDP gathering, ownership routing or model reload. Full model validation must
also compare loaded parameters or logits against a reference, repeat updates,
exercise EP/M:N placement and ring reuse, and test a failed consumer. No
throughput or numerical-equivalence claim is made by the CPU tests.

## Failure and shutdown

The default READ deadline is 300 seconds from the start of completion polling.
Producer stall detection remains governed by `ShardedRDTTrainerInitInfo`.
Failed or timed-out READs poison the session and retain registered storage;
terminate the affected worker instead of retrying or switching transports.

Finish all updates, shut down inference transfer engines first (drain scatter
and quant queues, disconnect HIXL, deregister receive memory, release consumer
leases), then shut down trainer engines. Producers reject shutdown while
consumer leases remain, so an incoming READ cannot race source deregistration.
A crashed consumer leaves a lease: terminate the failed job's own processes
through its normal lifecycle. Do not clear that lease while a reader may live.

Registration, disconnect and deregistration failures propagate to the caller.
Do not catch them and continue reusing the same tensors. HIXL errors include
the failing operation and status; size mismatches fail before the READ.
