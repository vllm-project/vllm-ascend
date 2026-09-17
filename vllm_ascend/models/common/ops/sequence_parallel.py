# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    """Shard the token axis across TP ranks for sequence parallelism.

    The pad+chunk shape math must stay opaque to dynamo: a plain Python
    implementation gets baked into the compiled graph with the trace-time
    shape, so graph capture at any other bucket size shards to a wrong row
    count (input_ids keeps the warmup shard size while hidden states use the
    real one, crashing the hash router with ``input_ids.numel() != x.rows``).
    Route through the registered upstream ``sequence_parallel_chunk_impl``
    whose fake impl derives ``cdiv(num_tokens, tp)`` dynamically. That op
    pads the last dim only, so 1D inputs (input_ids, token_mask) go through
    a [T, 1] view.
    """
    if x.ndim == 1:
        return sequence_parallel_chunk(x.reshape(-1, 1)).reshape(-1)
    return sequence_parallel_chunk(x)


def _ascend_sp_reduce_scatter_impl(x: torch.Tensor) -> torch.Tensor:
    """Pad rows to the TP multiple, then reduce-scatter across TP ranks.

    Wrapped in a custom op so the modulo padding stays invisible to dynamo
    (same shape-baking hazard as ``sp_shard``).
    """
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    pad_shape = [sp_pad, x.shape[1]]
    x = torch.cat([x, x.new_zeros(pad_shape)], dim=0)
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def _ascend_sp_reduce_scatter_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    return torch.empty(
        [cdiv(x.shape[0], tp_size), x.shape[1]],
        dtype=x.dtype,
        device=x.device,
    )


direct_register_custom_op(
    op_name="ascend_sp_reduce_scatter_impl",
    op_func=_ascend_sp_reduce_scatter_impl,
    mutates_args=[],
    fake_impl=_ascend_sp_reduce_scatter_fake,
    dispatch_key="PrivateUse1",
)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    return torch.ops.vllm.ascend_sp_reduce_scatter_impl(x)


def _ascend_sp_padding_mask_impl(is_padding: torch.Tensor) -> torch.Tensor:
    """Pad with True rows up to the TP multiple, then take this rank's chunk.

    The output row layout matches ``sp_shard`` so the mask stays aligned with
    the sharded hidden states. A custom op keeps the modulo padding invisible
    to dynamo (same shape-baking hazard as ``sp_shard``).
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-is_padding.shape[0]) % tp_size
    is_padding = torch.cat([is_padding, is_padding.new_ones((sp_pad,))], dim=0)
    chunk = is_padding.shape[0] // tp_size
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]


def _ascend_sp_padding_mask_fake(is_padding: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    return torch.empty(
        [cdiv(is_padding.shape[0], tp_size)],
        dtype=is_padding.dtype,
        device=is_padding.device,
    )


direct_register_custom_op(
    op_name="ascend_sp_padding_mask_impl",
    op_func=_ascend_sp_padding_mask_impl,
    fake_impl=_ascend_sp_padding_mask_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens

    return torch.ops.vllm.ascend_sp_padding_mask_impl(is_padding)
