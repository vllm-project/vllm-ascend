import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.forward_context import get_forward_context
from vllm.utils import direct_register_custom_op

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.utils import npu_stream_switch, prefetch_stream


def _maybe_all_gather_and_maybe_unpad_impl(
        x: torch.Tensor,
        label: bool,
        is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x

    sp_enabled = forward_context.sp_enabled
    if sp_enabled and label:
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is None or not is_ep_comm:
            x = tensor_model_parallel_all_gather(x, 0)
            pad_size = forward_context.pad_size
            if pad_size > 0:
                x = x[:-pad_size, :]
        else:
            x = get_ep_group().all_gather(x, 0)
            # unpad
            num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
            result = torch.empty(
                (num_tokens_across_dp_cpu.sum(), *x.shape[1:]),
                device=x.device,
                dtype=x.dtype)
            dp_size = get_dp_group().world_size
            x = x.view(dp_size, forward_context.padded_length, *x.shape[1:])
            offset = 0
            for idx in range(dp_size):
                num_tokens_dp = num_tokens_across_dp_cpu[idx]
                result[offset:offset +
                       num_tokens_dp, :] = x[idx, :num_tokens_dp, :]
                offset += num_tokens_dp
            x = result

    return x


def _maybe_pad_and_reduce_impl(x: torch.Tensor,
                               is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    if not forward_context.sp_enabled:
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        pad_size = forward_context.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        # padding
        dp_size = get_dp_group().world_size
        num_tokens_across_dp_cpu = \
            get_forward_context().dp_metadata.num_tokens_across_dp_cpu
        padded_x = torch.empty(
            (dp_size, forward_context.padded_length, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype)
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            padded_x[idx, :num_tokens_dp] = x[offset:offset + num_tokens_dp]
            offset += num_tokens_dp

        return get_ep_group().reduce_scatter(padded_x.view(-1, *x.shape[1:]),
                                             0)


def _maybe_prefetch_mlp_gate_up_proj_impl(x_dependency: torch.Tensor,
                                          prefix: str) -> None:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return

    if not forward_context.prefetch_mlp_enabled:
        return
    model_instance = forward_context.model_instance
    prefetch_stream = forward_context.prefetch_stream
    layer_idx = int(prefix.split('.')[2])

    # start point of gate_up_proj weight prefetch
    if prefix.split('.')[-2] == "self_attn":
        forward_context.prefetch_mlp_gate_up_proj = True
    if forward_context.prefetch_mlp_gate_up_proj:
        prefetch_stream.wait_stream(torch.npu.current_stream())

        with torch.npu.stream(prefetch_stream):
            mlp_gate_up_prefetch_size = envs_ascend.VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE
            torch_npu.npu_prefetch(model_instance.model.layers[layer_idx].mlp.gate_up_proj.weight, \
                                x_dependency, mlp_gate_up_prefetch_size)
    return


def _maybe_all_gather_and_maybe_unpad_fake(
        x: torch.Tensor,
        label: bool,
        is_ep_comm: bool = False) -> torch.Tensor:

    if get_forward_context().sp_enabled and label:
        return torch.empty(
            (x.shape[0] * get_tensor_model_parallel_world_size(),
             *x.shape[1:]),
            device=x.device,
            dtype=x.dtype)

    return x


def _maybe_pad_and_reduce_fake(x: torch.Tensor,
                               is_ep_comm: bool = False) -> torch.Tensor:
    if get_forward_context().sp_enabled:
        return torch.empty(
            (x.shape[0] // get_tensor_model_parallel_world_size(),
             *x.shape[1:]),
            device=x.device,
            dtype=x.dtype)

    return x


def _maybe_prefetch_mlp_gate_up_proj_impl_fake(x_dependency: torch.Tensor,
                                               prefix: str) -> None:
    return


def _maybe_prefetch_mlp_down_proj_impl(x_dependency: torch.Tensor) -> None:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return

    if not forward_context.prefetch_mlp_enabled:
        return
    forward_context.prefetch_mlp_down_proj = True
    model_instance = forward_context.model_instance
    prefetch_stream = forward_context.prefetch_stream
    layer_idx = forward_context.layer_idx

    # start point of down_proj weight prefetch
    prefetch_stream.wait_stream(torch.npu.current_stream())

    with torch.npu.stream(prefetch_stream):
        mlp_down_prefetch_size = envs_ascend.VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE
        torch_npu.npu_prefetch(model_instance.model.layers[layer_idx].mlp.down_proj.weight, \
                            x_dependency, mlp_down_prefetch_size)
    forward_context.layer_idx += 1
    return


def _maybe_prefetch_mlp_down_proj_impl_fake(
        x_dependency: torch.Tensor) -> None:
    return


def _maybe_wait_prefetch_done_impl(x: torch.Tensor) -> None:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return

    if not forward_context.prefetch_mlp_enabled:
        return
    if forward_context.prefetch_mlp_gate_up_proj or \
        forward_context.prefetch_mlp_down_proj:
        prefetch_stream = forward_context.prefetch_stream
        # wait until prefetch done
        torch.npu.current_stream().wait_stream(prefetch_stream)
        forward_context.prefetch_mlp_gate_up_proj = False
        forward_context.prefetch_mlp_down_proj = False
    return


def _maybe_wait_prefetch_done_impl_fake(x: torch.Tensor) -> None:
    return


def _prefetch_preprocess_impl(weight: torch.Tensor, start_flag: torch.Tensor,
                              max_weight_size: int) -> None:
    calculation_stream = torch_npu.npu.current_stream()
    weight_prefetch_stream = prefetch_stream()
    weight_prefetch_stream.wait_stream(calculation_stream)
    with npu_stream_switch(weight_prefetch_stream):
        maybe_npu_prefetch(inputs=weight,
                           dependency=start_flag,
                           max_size=max_weight_size)


def _prefetch_preprocess_impl_fake(weight: torch.Tensor,
                                   start_flag: torch.Tensor,
                                   max_weight_size: int) -> None:
    return


def _prefetch_postprocess_impl(stop_flag: torch.Tensor) -> None:
    calculation_stream = torch_npu.npu.current_stream()
    weight_prefetch_stream = prefetch_stream()
    calculation_stream.wait_stream(weight_prefetch_stream)


def _prefetch_postprocess_impl_fake(stop_flag: torch.Tensor) -> None:
    return


def _maybe_all_reduce_tensor_model_parallel_impl(
        final_hidden_states: torch.Tensor) -> torch.Tensor:
    forward_context = get_forward_context()
    moe_comm_type = forward_context.moe_comm_type
    if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2
                         } or forward_context.sp_enabled:
        return final_hidden_states
    else:
        return tensor_model_parallel_all_reduce(final_hidden_states)


direct_register_custom_op(op_name="maybe_all_gather_and_maybe_unpad",
                          op_func=_maybe_all_gather_and_maybe_unpad_impl,
                          fake_impl=_maybe_all_gather_and_maybe_unpad_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_pad_and_reduce",
                          op_func=_maybe_pad_and_reduce_impl,
                          fake_impl=_maybe_pad_and_reduce_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_prefetch_mlp_gate_up_proj",
                          op_func=_maybe_prefetch_mlp_gate_up_proj_impl,
                          fake_impl=_maybe_prefetch_mlp_gate_up_proj_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_prefetch_mlp_down_proj",
                          op_func=_maybe_prefetch_mlp_down_proj_impl,
                          fake_impl=_maybe_prefetch_mlp_down_proj_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_wait_prefetch_done",
                          op_func=_maybe_wait_prefetch_done_impl,
                          fake_impl=_maybe_wait_prefetch_done_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="prefetch_preprocess",
                          op_func=_prefetch_preprocess_impl,
                          fake_impl=_prefetch_preprocess_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="prefetch_postprocess",
                          op_func=_prefetch_postprocess_impl,
                          fake_impl=_prefetch_postprocess_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="maybe_all_reduce_tensor_model_parallel",
                          op_func=_maybe_all_reduce_tensor_model_parallel_impl,
                          fake_impl=lambda x: x,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")
