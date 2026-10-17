# Sparse weight transfer over HCCL

Configure the inference workers with `WeightTransferConfig(backend="sparse_hccl")`.
The model must already contain a known baseline. Updates use full checkpoint
names, shapes and flat `int32` indices; values replace existing positions.

The Ascend worker and trainer inherit vLLM's sparse NCCL engines. They reuse
`SparseWeightPatch`, update metadata validation, the no-op sparse reload
lifecycle, sender selection and the one-shot/chunk interfaces. Ascend overrides
NCCL initialization, CUDA stream operations and communicator cleanup. Worker
patches are applied by vLLM's `load_checkpoint_weight_patches`; this adapter
does not implement a second NaN-masked loader.

## Trainer interface

```python
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLTrainerInitInfo,
)

engine = WeightTransferTrainerFactory.trainer_init(
    SparseHCCLTrainerInitInfo(
        master_address=address,
        master_port=port,
        world_size=number_of_workers + 1,
        rank=trainer_rank,
    ),
    client=client,
)
engine.send_weights(patches)  # Owns start, send and finish.
```

Only trainer rank zero joins the transfer group. Patch indices and values must
be contiguous tensors on the communicator's NPU. For a caller-owned session,
call `client.start_weight_update()`, send one or more
`engine.send_weight_chunk(patches)` calls and then call
`client.finish_weight_update(weight_version)`. The caller must coordinate these
control-plane calls on its designated sender. A send returns after receiver
application and synchronization of the sending stream.

## Limits and failure handling

The default wire sequence follows sparse NCCL: indices and then values for each
patch. `SparseHCCLTrainerInitInfo(packed=True)` optionally batches consecutive
patches of the same dtype. This preserves input order, including repeated names.
For preassembled buffers, use `send_packed_chunk(update_info, indices, values)`
with `SparseHCCLWeightTransferUpdateInfo(packed=True)`. Its metadata must match
the tensor lengths and dtype. The call owns no start/finish lifecycle. An
optional `torch_hash_v1` checksum is checked before applying a packed payload;
sender and receiver must provide the same supported hash implementation.

`direct_moe=True` enables guarded Qwen3-MoE writes. Supported unquantized ND
weights use global checkpoint expert IDs mapped to local EP slots and actual
MoE TP coordinates. Dynamic EPLB, unsupported layouts and repeated names retain
the public loader fallback. The complete local patch list is checked for NaNs,
bounds and duplicate indices before any direct write. The optimization saves
full checkpoint staging for supported expert weights; it does not remove the
memory needed by the live model or fallback patches. `verify=True` checks local
writes and introduces diagnostic synchronization.

If the caller needs an initial complete checkpoint transfer, initialize with
`dense_seed=True` and send packed metadata with `encoding="dense"`. That first
session uses the existing HCCL layerwise reload lifecycle, and each values
tensor is loaded through the public checkpoint loader. Finish the complete
dense session before starting sparse updates. Dense metadata is rejected after
the seed session; reconnect to establish a new baseline after a partial update.
A loader extension can override `_apply_checkpoint_patches`; it must validate
the complete input before writing.

The upstream loader accepts floating sparse values and rejects NaNs,
out-of-range indices and duplicate indices within a patch. Repeated patches
with the same name are applied in input order. Its supported loader contract
requires final same-shaped floating-point `Tensor.copy_` writes; composed,
intermediate or custom-write loaders require separate validation. Fallback
loading creates full checkpoint-shaped staging tensors, so staging memory is
not proportional to sparse payload size.

Sparse updates do not support draft-model updates or rollback. Keep generation
paused after a potentially partial update until a known baseline has been
restored or the affected workers restarted. Checking an already failed RPC
before broadcasting does not provide a timeout for a peer that fails later.

## Hardware regression

On three free NPUs, run:

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9,10 \
  torchrun --standalone --nproc-per-node=3 \
  tests/e2e/multi_card/test_sparse_hccl_transfer.py
```

The example device IDs select three physical NPUs on the validation host;
replace them with free device IDs on your host. The regression uses one trainer
and two workers, six BF16 update chunks over the standard and packed paths, and repeated checkpoint names. Each
worker compares its complete shard, including untouched elements, byte for byte
after every chunk.

An isolated Qwen3-30B-A3B integration run also completed on eight NPUs with
TP2/PP2/EP2, a dense seed and three subsequent sparse updates. Steady update wall
times were 50.6, 62.1 and 44.5 seconds with checksum and local verification enabled.
These include training export/gather and the complete update lifecycle; they
are not receiver apply times or evidence of an end-to-end speed improvement.
