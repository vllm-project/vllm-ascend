import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.rotary_embedding import rope_forward_oot
from vllm_ascend.ops.triton.muls_add import muls_add_triton
from vllm_ascend.utils import enable_sp_by_pass, is_vl_model


def _maybe_chunk_residual_impl(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    try:
        get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) != residual.size(0):
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            residual = F.pad(residual, (0, 0, 0, pad_size))
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]

    return residual


def _maybe_all_gather_and_maybe_unpad_impl(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x

    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)
    if flash_comm_v1_enabled and label:
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is None or not is_ep_comm:
            x = tensor_model_parallel_all_gather(x, 0)
            pad_size = _EXTRA_CTX.pad_size
            if pad_size > 0:
                x = x[:-pad_size]
        else:
            x = get_ep_group().all_gather(x, 0)
            if enable_sp_by_pass():  # TODO: do unpad
                return x
            # unpad
            num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
            result = torch.empty((num_tokens_across_dp_cpu.sum(), *x.shape[1:]), device=x.device, dtype=x.dtype)
            dp_size = get_dp_group().world_size
            x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
            offset = 0
            for idx in range(dp_size):
                num_tokens_dp = num_tokens_across_dp_cpu[idx]
                result[offset : offset + num_tokens_dp] = x[idx, :num_tokens_dp]
                offset += num_tokens_dp
            x = result

    return x


def _maybe_pad_and_reduce_impl(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)

    if not flash_comm_v1_enabled or (_EXTRA_CTX.is_draft_model and is_vl_model() and not is_ep_comm):
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        if enable_sp_by_pass():
            return get_ep_group().reduce_scatter(x.view(-1, *x.shape[1:]), 0)
        # padding
        dp_size = get_dp_group().world_size
        num_tokens_across_dp_cpu = get_forward_context().dp_metadata.num_tokens_across_dp_cpu
        padded_x = torch.empty((dp_size, _EXTRA_CTX.padded_length, *x.shape[1:]), device=x.device, dtype=x.dtype)
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            padded_x[idx, :num_tokens_dp] = x[offset : offset + num_tokens_dp]
            offset += num_tokens_dp

        return get_ep_group().reduce_scatter(padded_x.view(-1, *x.shape[1:]), 0)


def _maybe_all_gather_and_maybe_unpad_fake(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    if _EXTRA_CTX.flash_comm_v1_enabled and label:
        return torch.empty(
            (x.shape[0] * get_tensor_model_parallel_world_size(), *x.shape[1:]), device=x.device, dtype=x.dtype
        )

    return x


def _maybe_pad_and_reduce_fake(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    if _EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm):
        return torch.empty(
            (x.shape[0] // get_tensor_model_parallel_world_size(), *x.shape[1:]), device=x.device, dtype=x.dtype
        )

    return x


def _maybe_all_reduce_tensor_model_parallel_impl(final_hidden_states: torch.Tensor) -> torch.Tensor:
    moe_comm_type = _EXTRA_CTX.moe_comm_type
    if (
        moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
        or _EXTRA_CTX.flash_comm_v1_enabled
    ):
        return final_hidden_states
    else:
        return tensor_model_parallel_all_reduce(final_hidden_states)


def _matmul_and_reduce_impl(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.custom_op is not None
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    output = self.custom_op.matmul_and_reduce(input_parallel, bias_)

    return output


def _matmul_and_reduce_impl_fake(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    num_tokens = input_parallel.size(0)
    if _EXTRA_CTX.flash_comm_v1_enabled:
        num_tokens = num_tokens // self.tp_size
    output = torch.empty(
        size=(num_tokens, self.output_size_per_partition), device=input_parallel.device, dtype=input_parallel.dtype
    )

    return output


# TODO(Angazenn): The reason why we use a custom op to encapsulate npu_quantize
# is that aclnnAscendQuantV3(npu_quantize) use div_mode=False, while
# aclnnAddRmsNormQuantV2(npu_add_rms_norm_quant) use div_moe=True. We have to
# pass input_scale and input_scale_reciprocal at the same time to avoid redundant
# reciprocal calculation in fussion pass. We shall remove this once
# aclnnAddRmsNormQuantV2 supports div_moe=False.
def _quantize_impl(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


def _quantize_impl_fake(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


def _dynamic_quant_matmul(
    input_: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(input_, dst_type=torch.int8)
    need_unsqueeze = False
    if pertoken_scale.dim() == 2:
        need_unsqueeze = True
        quantized_x = quantized_x.squeeze(dim=1)
        pertoken_scale = pertoken_scale.squeeze(dim=1)

    output = torch_npu.npu_quant_matmul(
        quantized_x,
        weight,
        weight_scale,
        pertoken_scale=pertoken_scale,
        bias=None,
        output_dtype=input_.dtype,
    )
    if need_unsqueeze:
        output = output.unsqueeze(dim=1)
    return output


def _all_gather_then_dynamic_quant_matmul(
    input_: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    gathered = tensor_model_parallel_all_gather(input_, 0)
    pad_size = _EXTRA_CTX.pad_size
    if pad_size > 0:
        gathered = gathered[:-pad_size]
    return _dynamic_quant_matmul(gathered, weight, weight_scale)


def _all_gather_then_unquantized_matmul(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    gathered = tensor_model_parallel_all_gather(input_, 0)
    pad_size = _EXTRA_CTX.pad_size
    if pad_size > 0:
        gathered = gathered[:-pad_size]
    return torch.ops.vllm.unquantized_gemm(gathered, weight, bias)


def _is_tp_all_gather_enabled() -> bool:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return False

    return bool(getattr(forward_context, "flash_comm_v1_enabled", False))


def _all_gather_matmul_comm_mode() -> str:
    device_type = get_ascend_device_type()
    if device_type == AscendDeviceType.A5:
        return "ccu"
    return "aiv"


def _get_tp_hcom_name() -> tuple[str, int]:
    tp_group = get_tp_group()
    hcom_name = tp_group.device_group._get_backend(torch.device("npu")).get_hccl_comm_name(
        tp_group.rank_in_group,
    )
    return hcom_name, tp_group.world_size


def _all_gather_dynamic_quant_matmul_impl(
    input_: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if not _is_tp_all_gather_enabled():
        return _dynamic_quant_matmul(input_, weight, weight_scale)

    if input_.dim() != 2:
        return _all_gather_then_dynamic_quant_matmul(input_, weight, weight_scale)

    if get_ascend_device_type() == AscendDeviceType.A5:
        return _all_gather_then_dynamic_quant_matmul(input_, weight, weight_scale)

    hcom_name, world_size = _get_tp_hcom_name()
    if world_size == 1:
        return _dynamic_quant_matmul(input_, weight, weight_scale)

    quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(input_, dst_type=torch.int8)
    x1_scale = pertoken_scale.reshape(-1, 1).contiguous()
    x2_scale = weight_scale.reshape(1, -1).to(torch.float32).contiguous()

    output, _ = DeviceOperator.npu_all_gather_base_mm(
        quantized_x,
        weight,
        hcom_name,
        world_size,
        bias=None,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        gather_index=0,
        gather_output=True,
        comm_turn=0,
        output_dtype=input_.dtype,
        comm_mode=_all_gather_matmul_comm_mode(),
    )

    pad_size = _EXTRA_CTX.pad_size
    if pad_size > 0:
        output = output[:-pad_size]
    return output


def _all_gather_unquantized_matmul_impl(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _is_tp_all_gather_enabled():
        return torch.ops.vllm.unquantized_gemm(input_, weight, bias)

    if input_.dim() != 2:
        return _all_gather_then_unquantized_matmul(input_, weight, bias)

    hcom_name, world_size = _get_tp_hcom_name()
    if world_size == 1:
        return torch.ops.vllm.unquantized_gemm(input_, weight, bias)

    output, _ = DeviceOperator.npu_all_gather_base_mm(
        input_,
        weight.t(),
        hcom_name,
        world_size,
        bias=bias,
        x1_scale=None,
        x2_scale=None,
        gather_index=0,
        gather_output=True,
        comm_turn=0,
        output_dtype=input_.dtype,
        comm_mode=_all_gather_matmul_comm_mode(),
    )

    pad_size = _EXTRA_CTX.pad_size
    if pad_size > 0:
        output = output[:-pad_size]
    return output


def _all_gather_dynamic_quant_matmul_fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    try:
        world_size = get_tensor_model_parallel_world_size()
        pad_size = _EXTRA_CTX.pad_size
    except Exception:
        world_size = 1
        pad_size = 0

    if not _is_tp_all_gather_enabled():
        world_size = 1
        pad_size = 0

    output_tokens = input_.shape[0] * world_size
    if pad_size > 0:
        output_tokens -= pad_size
    return torch.empty(
        (output_tokens, *input_.shape[1:-1], weight.shape[-1]),
        device=input_.device,
        dtype=input_.dtype,
    )


def _all_gather_unquantized_matmul_fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    try:
        world_size = get_tensor_model_parallel_world_size()
        pad_size = _EXTRA_CTX.pad_size
    except Exception:
        world_size = 1
        pad_size = 0

    if not _is_tp_all_gather_enabled():
        world_size = 1
        pad_size = 0

    output_tokens = input_.shape[0] * world_size
    if pad_size > 0:
        output_tokens -= pad_size
    return torch.empty(
        (output_tokens, *input_.shape[1:-1], weight.shape[0]),
        device=input_.device,
        dtype=input_.dtype,
    )


def _rope_forward_oot_impl_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


def _muls_add_impl_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="maybe_chunk_residual",
    op_func=_maybe_chunk_residual_impl,
    fake_impl=lambda x, residual: torch.empty_like(x),
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_all_gather_and_maybe_unpad",
    op_func=_maybe_all_gather_and_maybe_unpad_impl,
    fake_impl=_maybe_all_gather_and_maybe_unpad_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_pad_and_reduce",
    op_func=_maybe_pad_and_reduce_impl,
    fake_impl=_maybe_pad_and_reduce_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_all_reduce_tensor_model_parallel",
    op_func=_maybe_all_reduce_tensor_model_parallel_impl,
    fake_impl=lambda x: x,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="matmul_and_reduce",
    op_func=_matmul_and_reduce_impl,
    fake_impl=_matmul_and_reduce_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="quantize",
    op_func=_quantize_impl,
    fake_impl=_quantize_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="all_gather_dynamic_quant_matmul",
    op_func=_all_gather_dynamic_quant_matmul_impl,
    fake_impl=_all_gather_dynamic_quant_matmul_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="all_gather_unquantized_matmul",
    op_func=_all_gather_unquantized_matmul_impl,
    fake_impl=_all_gather_unquantized_matmul_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="npu_rotary_embedding",
    op_func=rope_forward_oot,
    fake_impl=_rope_forward_oot_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="muls_add",
    op_func=muls_add_triton,
    fake_impl=_muls_add_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
