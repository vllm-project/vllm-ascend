from typing import Optional
import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.utils import direct_register_custom_op

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.utils import npu_stream_switch, prefetch_stream


def _apply_matmul_and_reduce_impl(
        prefix: str,
        input_parallel: torch.Tensor,
    ) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
        sp_enabled = forward_context.sp_enabled
    except AssertionError:
        sp_enabled = False

    layer_idx = int(prefix.split('.')[2])
    layer_name = prefix.split('.')[-1]
    model_instance = forward_context.model_instance
    if layer_name == "o_proj":
        layer = model_instance.model.layers[layer_idx].self_attn.o_proj
    if layer_name == "down_proj":
        layer = model_instance.model.layers[layer_idx].mlp.down_proj
    if not sp_enabled:
        output_parallel = layer.quant_method.apply(layer,
                                                   input_parallel,
                                                   bias=None)
        return tensor_model_parallel_all_reduce(output_parallel)

    pad_size = forward_context.pad_size
    if pad_size > 0:
        input_parallel = F.pad(input_parallel, (0, 0, 0, pad_size))

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod, quant_per_tensor
    # no quant
    if isinstance(layer.quant_method, UnquantizedLinearMethod):
        output_parallel = torch.empty(input_parallel.shape[0] // layer.tp_size,
                                      layer.weight.shape[0],
                                      dtype=layer.params_dtype,
                                      device=input_parallel.device)
        hcom_name = get_tp_group().device_group._get_backend(torch.device('npu')).get_hccl_comm_name(layer.tp_rank)
        world_size = layer.tp_size
        comm_mode = "aiv"
        output = torch_npu.npu_mm_reduce_scatter_base(input_parallel, 
                                                      layer.weight.t(), 
                                                      hcom_name, 
                                                      world_size, 
                                                      reduce_op="sum", 
                                                      bias=None, 
                                                      comm_turn=0, 
                                                      comm_mode=comm_mode)
    # w8a8 quant
    elif isinstance(layer.quant_method.quant_method, AscendW8A8LinearMethod):
        if input_parallel.dtype != torch.int8:
            input_parallel_quant = quant_per_tensor(input_parallel, layer.aclnn_input_scale_reciprocal, layer.aclnn_input_offset)
        else:
            input_parallel_quant = input_parallel
        output_parallel = torch.empty(input_parallel_quant.shape[0] // layer.tp_size,
                                      layer.weight.shape[1],
                                      dtype=layer.params_dtype,
                                      device=input_parallel.device)
        quant_bias = layer.quant_bias
        hcom_name = get_tp_group().device_group._get_backend(torch.device('npu')).get_hccl_comm_name(layer.tp_rank)
        world_size = layer.tp_size
        deq_scale = layer.deq_scale
        output_dtype = torch.bfloat16
        comm_mode = "aiv"
        output_parallel = torch_npu.npu_mm_reduce_scatter_base(input_parallel_quant, 
                                                               layer.weight, 
                                                               hcom_name, 
                                                               world_size, 
                                                               reduce_op="sum", 
                                                               bias=None, 
                                                               comm_turn=0, 
                                                               x2_scale=deq_scale, 
                                                               output_dtype=output_dtype, 
                                                               comm_mode=comm_mode)
        output = torch.add(output_parallel, torch.mul(quant_bias, deq_scale).to(layer.params_dtype))
    else:
        output_parallel = layer.quant_method.apply(layer,
                                                   input_parallel,
                                                   bias=None)
        output = tensor_model_parallel_reduce_scatter(output_parallel, 0)

    return output


def _apply_matmul_and_reduce_impl_fake(
        prefix: str,
        input_parallel: torch.Tensor,
    ) -> torch.Tensor:
    model_instance = get_forward_context().model_instance
    layer_idx = int(prefix.split('.')[2])
    layer_name = prefix.split('.')[-1]
    if layer_name == "o_proj":
        layer = model_instance.model.layers[layer_idx].self_attn.o_proj
    if layer_name == "down_proj":
        layer = model_instance.model.layers[layer_idx].mlp.down_proj
    output_size_per_partition = layer.output_size_per_partition
    pad_size = output_size_per_partition - input_parallel.size(1)
    output = F.pad(input_parallel, (0, pad_size))

    return output

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
        x = tensor_model_parallel_all_gather(x, 0)
        pad_size = forward_context.pad_size
        if pad_size > 0:
            x = x[:-pad_size, :]
    return x


def _maybe_pad_and_reduce_impl(x: torch.Tensor) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    sp_enabled = forward_context.sp_enabled
    if sp_enabled:
        pad_size = forward_context.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        return tensor_model_parallel_all_reduce(x)


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
    if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2}:
        return final_hidden_states
    else:
        return tensor_model_parallel_all_reduce(final_hidden_states)


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

direct_register_custom_op(op_name="apply_matmul_and_reduce",
                          op_func=_apply_matmul_and_reduce_impl,
                          fake_impl=_apply_matmul_and_reduce_impl_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")
