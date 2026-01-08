import torch
from typing import Optional
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

@triton.heuristics(
    {"HAS_BIAS": lambda args: args["norm_bias_ptr"] is not None})
@triton.jit
def add_rmsnorm_bias_kernel(input_ptr, residual_ptr, norm_weight_ptr,
                            norm_bias_ptr, output_ptr, output2_ptr, batch_size,
                            hidden_size: tl.constexpr, eps: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr):
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


def add_rmsnorm_bias(input: torch.Tensor, residual: torch.Tensor,
                     norm_weight: torch.Tensor,
                     norm_bias: Optional[torch.Tensor],
                     eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    input = input.contiguous()
    if residual is not None:
        residual = residual.contiguous()
    norm_weight = norm_weight.contiguous()
    if norm_bias is not None:
        norm_bias = norm_bias.contiguous()
    num_vectorcore = get_vectorcore_num()
    batch_size = input.shape[0]
    hidden_size = input.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    n_rows = min(batch_size, num_vectorcore)
    output = torch.empty(batch_size,
                         hidden_size,
                         device=input.device,
                         dtype=input.dtype)
    output2 = torch.empty(batch_size,
                          hidden_size,
                          device=input.device,
                          dtype=input.dtype)
    add_rmsnorm_bias_kernel[(n_rows, 1, 1)](input, residual, norm_weight,
                                            norm_bias, output, output2,
                                            batch_size, hidden_size, eps,
                                            BLOCK_SIZE)
    return output, output2
