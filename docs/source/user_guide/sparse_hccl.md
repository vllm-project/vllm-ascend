# Checkpoint-coordinate sparse HCCL updates

The `sparse_hccl` weight-transfer backend sends replacement values and flat
`int32` indices in checkpoint coordinates. Every inference worker receives the
same patches. The model's native `load_weights()` handles tensor-parallel
sharding, expert placement, packed QKV, vocabulary partitions, and fused experts.

Use a shared, initialized baseline on trainer and inference workers. Patch names
and `full_shape` describe full checkpoint tensors, rather than local runtime
parameters. Indices must be unique within each patch and inside its full shape.
NaN replacement values are rejected because NaN marks unchanged elements in the
upstream checkpoint-patch loader. Repeated patches can update the same checkpoint
tensor in input order.

## Configuration

Start inference with `--weight-transfer-config '{"backend":"sparse_hccl"}'`
and `--additional-config '{"weight_nz_mode":0}'`. One trainer sender joins rank 0;
all inference workers join with `rank_offset=1`. For PP=1, the transfer group
size is `1 + TP * DP`, including all DP replicas. TP4/DP1/EP4 needs five devices;
TP4/DP2/EP8 needs nine. Every participant must use a distinct device.

Only unquantized floating-point weights, PP=1, and static expert placement
(EPLB disabled), and ND tensor storage are supported. Fused MC2 expert lists
and NZ weights are rejected before transfer. Ascend's unquantized expert method
temporarily exposes checkpoint-loader views of the transposed GMM parameters;
updates retain their inference storage and restore the layout even on failure.
Draft-model updates and packed sparse transport
are unsupported. The upstream sparse-loader contract requires a final,
same-shaped floating-point `Tensor.copy_`; composed loaders and custom writes
need separate validation before use.

## Trainer API

```python
import torch
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLTrainerInitInfo,
    SparseWeightPatch,
)
from vllm_ascend.distributed.weight_transfer import register_engine

register_engine()  # Standalone trainers also need Ascend factory registration.
client = HTTPVLLMWeightSyncClient("http://127.0.0.1:8000")
engine = WeightTransferTrainerFactory.trainer_init(
    init_info=SparseHCCLTrainerInitInfo(
        master_address="127.0.0.1",
        master_port=29501,
        world_size=5,
        rank=0,
    ),
    client=client,
)
patches = [
    SparseWeightPatch(
        name="model.layers.0.self_attn.q_proj.weight",
        full_shape=(4096, 4096),
        indices=torch.tensor([0, 4096], dtype=torch.int32),
        values=torch.tensor([0.1, 0.2], dtype=torch.bfloat16),
    ),
]
engine.send_weights(patches)
engine.shutdown()
```

Pause generation before updating and resume after successful completion.
`send_weights()` owns one start/update/finish lifecycle. To bound transfer
buffers, the caller can own a lifecycle across multiple chunks:

```python
client.start_weight_update()
for patches in patch_chunks:
    engine.send_weight_chunk(patches)
client.finish_weight_update()
```

Empty patch lists are no-ops. Noncontiguous or CPU patch tensors are moved to
the trainer communicator's device before the worker lifecycle starts. The wire
payload is O(nnz), but full checkpoint-shaped NaN staging and local application
remain O(N). Chunk bounds must account for checkpoint shapes, not only sparse
payload bytes. A single tensor can exceed the upstream loader's batching target.

Updates are in-place and nontransactional. A failed update must remain paused
until workers are restarted or restored from a known dense baseline. The trainer
does not call finish after a failed transfer.

## Validation

`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py` covers validation,
rank mapping, lifecycle, and teardown. The real-model E2E uses a deterministic
synthetic, one-layer Qwen3 MoE checkpoint with eight experts:

```bash
pytest -s tests/e2e/pull_request/rlhf/test_sparse_hccl_checkpoint.py
```

It compares every runtime parameter hash on every worker against native dense
checkpoint loading, then compares greedy generation. It covers attention
projections, router, every expert's gate/up/down projection, embeddings, LM head,
and norms in TP4/EP4 and TP4/DP2/EP8. These tests check mapping correctness;
they do not establish pretrained-model accuracy or production performance.
