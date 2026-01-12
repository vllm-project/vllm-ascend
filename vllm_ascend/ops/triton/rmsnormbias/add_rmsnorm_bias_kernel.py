import torch

from typing import Optional
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

@triton.heuristics(
    {"HAS_BIAS": lambda args: args["norm_bias_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["residual_ptr"] is not None})
@triton.jit
def add_rmsnorm_bias_kernel(input_ptr, residual_ptr, norm_weight_ptr,
                            norm_bias_ptr, output_ptr, output2_ptr, batch_size,
                            hidden_size: tl.constexpr, eps: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr,
                            HAS_Z: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    valid_mask = cols < hidden_size
    norm_weight_values = tl.load(norm_weight_ptr + cols,
                                 mask=valid_mask,
                                 other=0.0)
    input_offsets = row_start * hidden_size + cols
    for _ in tl.range(row_start, batch_size, row_step):
        # add
        buffered_values = tl.load(input_ptr + input_offsets,
                                  mask=valid_mask,
                                  other=0.0)
        if HAS_Z:
            buffered_values += tl.load(residual_ptr + input_offsets,
                                       mask=valid_mask,
                                       other=0.0)
        tl.store(output2_ptr + input_offsets, buffered_values, mask=valid_mask)
        buffered_values = buffered_values.to(tl.float32)
        # rmsnorm
        squares = buffered_values * buffered_values
        variance = tl.sum(squares) / hidden_size
        reciprocal_std = 1 / tl.sqrt(variance + eps)
        buffered_values = buffered_values * reciprocal_std
        buffered_values = buffered_values * norm_weight_values
        # add bias
        if HAS_BIAS:
            norm_bias_values = tl.load(norm_bias_ptr + cols,
                                       mask=valid_mask,
                                       other=0.0)
            buffered_values = buffered_values + norm_bias_values
        tl.store(output_ptr + input_offsets, buffered_values, mask=valid_mask)

        input_offsets += row_step * hidden_size


def add_rmsnorm_bias(
        input: torch.Tensor, residual: Optional[torch.Tensor],
        norm_weight: torch.Tensor, norm_bias: Optional[torch.Tensor],
        eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = input.shape
    is_3d = input.ndim == 3
    if is_3d:
        input_flat = input.reshape(-1, input.shape[-1])
        residual_flat = residual.reshape(-1, residual.shape[-1]) if residual is not None else None
    else:
        input_flat = input
        residual_flat = residual
    input_flat = input_flat.contiguous()
    if residual_flat is not None:
        residual_flat = residual_flat.contiguous()
    norm_weight = norm_weight.contiguous()
    if norm_bias is not None:
        norm_bias = norm_bias.contiguous()
    num_vectorcore = get_vectorcore_num()
    batch_size = input_flat.shape[0]
    hidden_size = input_flat.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    n_rows = min(batch_size, num_vectorcore)
    output_flat = torch.empty(batch_size,
                              hidden_size,
                              device=input.device,
                              dtype=input.dtype)
    output2_flat = torch.empty(batch_size,
                               hidden_size,
                               device=input.device,
                               dtype=input.dtype)
    add_rmsnorm_bias_kernel[(n_rows, 1, 1)](input_flat, residual_flat, norm_weight,
                                            norm_bias,output_flat, output2_flat,
                                            batch_size, hidden_size, eps,
                                            BLOCK_SIZE)

    if is_3d:
        output = output_flat.reshape(original_shape)
        output2 = output2_flat.reshape(original_shape)
    else:
        output = output_flat
        output2 = output2_flat

    return output, output2