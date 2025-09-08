import torch
import torch.nn.functional as F
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.forward_context import get_forward_context
from vllm.utils import direct_register_custom_op


def _maybe_chunk_residual_impl(x: torch.Tensor,
                               residual: torch.Tensor) -> torch.Tensor:
    if get_forward_context().flashcomm_v1_enabled:
        pad_size = get_forward_context().pad_size
        if pad_size > 0:
            residual = F.pad(residual, (0, 0, 0, pad_size))
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]

    return residual


def _maybe_all_gather_and_maybe_unpad_impl(x: torch.Tensor,
                                           label: bool) -> torch.Tensor:
    flashcomm_v1_enabled = get_forward_context().flashcomm_v1_enabled
    if flashcomm_v1_enabled and label:
        x = tensor_model_parallel_all_gather(x, 0)
        pad_size = get_forward_context().pad_size
        if pad_size > 0:
            x = x[:-pad_size, :]
    return x


def _maybe_pad_and_reduce_impl(x: torch.Tensor) -> torch.Tensor:
    flashcomm_v1_enabled = get_forward_context().flashcomm_v1_enabled
    if flashcomm_v1_enabled:
        pad_size = get_forward_context().pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        return tensor_model_parallel_all_reduce(x)


direct_register_custom_op(op_name="maybe_chunk_residual",
                          op_func=_maybe_chunk_residual_impl,
                          fake_impl=lambda x, residual: residual,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_all_gather_and_maybe_unpad",
                          op_func=_maybe_all_gather_and_maybe_unpad_impl,
                          fake_impl=lambda x, label: x,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_pad_and_reduce",
                          op_func=_maybe_pad_and_reduce_impl,
                          fake_impl=lambda x: x,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")