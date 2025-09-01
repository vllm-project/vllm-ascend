import torch
from vllm.utils import direct_register_custom_op
from vllm.distributed import tensor_model_parallel_all_gather, tensor_model_parallel_reduce_scatter, tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context


def _flashcomm_residual_chunk_impl(residual: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    flashcomm_v1_enabled = get_forward_context().flashcomm_v1_enabled
    if flashcomm_v1_enabled:
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]
    return residual


def _flashcomm_all_gather_impl(hidden_states: torch.Tensor) -> torch.Tensor:
    flashcomm_v1_enabled = get_forward_context().flashcomm_v1_enabled
    if flashcomm_v1_enabled:
        return tensor_model_parallel_all_gather(hidden_states, 0)
    else:
        return hidden_states


def _flashcomm_all_gather_with_condition_impl(hidden_states: torch.Tensor, label: bool) -> torch.Tensor:
    flashcomm_v1_enabled = get_forward_context().flashcomm_v1_enabled
    if flashcomm_v1_enabled and label:
        return tensor_model_parallel_all_gather(hidden_states, 0)
    else:
        return hidden_states


def _flashcomm_reduce_impl(hidden_states: torch.Tensor) -> torch.Tensor:
    flashcomm_v1_enabled = get_forward_context().flashcomm_v1_enabled
    if flashcomm_v1_enabled:
        return tensor_model_parallel_reduce_scatter(hidden_states, 0)
    else:
        return tensor_model_parallel_all_reduce(hidden_states)


direct_register_custom_op(
    op_name="flashcomm_residual_chunk",
    op_func=_flashcomm_residual_chunk_impl,
    fake_impl=lambda residual, tp_size, tp_rank: residual,
    mutates_args=[],
    dispatch_key="PrivateUse1"
)

direct_register_custom_op(
    op_name="flashcomm_all_gather",
    op_func=_flashcomm_all_gather_impl,
    fake_impl=lambda hidden_states: hidden_states,
    mutates_args=[],
    dispatch_key="PrivateUse1"
)

direct_register_custom_op(
    op_name="flashcomm_all_gather_with_condition",
    op_func=_flashcomm_all_gather_with_condition_impl,
    fake_impl=lambda hidden_states, label: hidden_states,
    mutates_args=[],
    dispatch_key="PrivateUse1"
)

direct_register_custom_op(
    op_name="flashcomm_reduce",
    op_func=_flashcomm_reduce_impl,
    fake_impl=lambda hidden_states: hidden_states,
    mutates_args=[],
    dispatch_key="PrivateUse1"
)
