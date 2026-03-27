# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:
    from triton.runtime import driver


@triton.jit
def _attention_lse_update_kernel(
    out_ptr,
    lse_ptr,
    output_ptr,
    lse_final_ptr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuse attention outputs from multiple ranks using LSE (Log-Sum-Exp) merging.
    
    Algorithm:
        LSE_final = log(sum(exp(LSE_i)))
        O_final = sum(exp(LSE_i - LSE_final) * O_i)
    
    Input shapes:
        out_ptr: [N, S*H, D] - attention outputs from N ranks
        lse_ptr: [N, S*H] - log-sum-exp values from N ranks
    Output shape:
        output_ptr: [S*H, D] - merged attention output
        lse_final_ptr: [S*H] - final LSE values
    """
    pid = tl.program_id(0)
    total_elements = S * H
    num_programs = tl.num_programs(0)
    
    rows_per_program = (total_elements + num_programs - 1) // num_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, total_elements)
    
    for row_idx in range(start_row, end_row):
        max_lse = float("-inf")
        
        for n in range(N):
            lse_val = tl.load(lse_ptr + n * total_elements + row_idx)
            if lse_val > max_lse:
                max_lse = lse_val
        
        sum_exp_lse = 0.0
        for n in range(N):
            lse_val = tl.load(lse_ptr + n * total_elements + row_idx)
            sum_exp_lse = sum_exp_lse + tl.exp(lse_val - max_lse)
        
        lse_final = max_lse + tl.log(sum_exp_lse)
        if lse_final_ptr is not None:
            tl.store(lse_final_ptr + row_idx, lse_final)
        
        for col_offset in range(0, D, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < D
            
            sum_weighted = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            for n in range(N):
                lse_val = tl.load(lse_ptr + n * total_elements + row_idx)
                weight = tl.exp(lse_val - lse_final)
                
                out_offset = n * total_elements * D + row_idx * D
                out_vals = tl.load(out_ptr + out_offset + col_idx, mask=mask, other=0.0)
                sum_weighted = sum_weighted + weight * out_vals
            
            tl.store(output_ptr + row_idx * D + col_idx, sum_weighted, mask=mask)


def attention_lse_update_triton(
    out_list: list[torch.Tensor],
    lse_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge multiple attention outputs using LSE-based weighted sum.
    
    Args:
        out_list: List of attention outputs, each [S*H, D]
        lse_list: List of LSE values, each [S*H]
    
    Returns:
        output: Merged attention output [S*H, D]
        lse_final: Final LSE values [S*H]
    """
    assert len(out_list) == len(lse_list), "out_list and lse_list must have same length"
    assert len(out_list) > 0, "Lists must not be empty"
    
    N = len(out_list)
    S_H, D = out_list[0].shape
    
    out_stacked = torch.stack(out_list, dim=0).contiguous()
    lse_stacked = torch.stack(lse_list, dim=0).contiguous()
    
    output = torch.empty(S_H, D, dtype=torch.float32, device=out_list[0].device)
    lse_final = torch.empty(S_H, dtype=torch.float32, device=out_list[0].device)
    
    S = S_H
    H = 1
    
    BLOCK_SIZE = 64
    if HAS_TRITON:
        max_grid_size = driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )["num_vectorcore"]
    else:
        max_grid_size = 128
    
    grid = (min(S_H, max_grid_size),)
    
    _attention_lse_update_kernel[grid](
        out_stacked,
        lse_stacked,
        output,
        lse_final,
        S,
        H,
        D,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, lse_final


def fused_dycp_attention_update_with_mask(
    dycp_mask: torch.Tensor,
    attn_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    dycp_group,
) -> torch.Tensor:
    """
    DYCP attention update using a mask tensor to select requests.
    
    This is a fused Triton implementation that replaces the original
    _npu_update_dycp_attn_with_mask function.
    
    Args:
        dycp_mask: Boolean mask tensor of shape [total_reqs], where True indicates 
                   a DYCP request that needs to be processed
        attn_output: Attention output tensor [total_reqs, H, D]
        softmax_lse: Log-sum-exp tensor [total_reqs, H, 1]
        dycp_group: DYCP process group for collective operations
    
    Returns:
        Updated attention output for all requests [total_reqs, H, D]
    """
    assert dycp_mask.any(), "dycp_mask should have at least one True value"
    assert dycp_mask.dtype == torch.bool, "dycp_mask should be a boolean tensor"
    
    dycp_size = dycp_group.world_size
    total_reqs = attn_output.shape[0]
    H = attn_output.shape[1]
    D = attn_output.shape[2]
    
    num_dycp_reqs = dycp_mask.sum().item()
    S = num_dycp_reqs
    
    dycp_softmax_lse = softmax_lse.to(torch.float32)[:num_dycp_reqs]
    dycp_attn_output = attn_output.to(torch.float32)[:num_dycp_reqs]
    
    attn_out_lse = torch.cat([dycp_attn_output, dycp_softmax_lse], dim=-1)
    
    attn_out_lse = dycp_group.all_gather(attn_out_lse.contiguous(), dim=0)
    
    attn_out_lse = attn_out_lse.view(dycp_size, S, H, D + 1)
    
    out_flat, lse_flat = torch.split(attn_out_lse, [D, 1], dim=-1)
    out_flat = out_flat.flatten(1, 2)
    lse_flat = lse_flat.flatten(1, -1)
    
    out_list = out_flat.unbind(0)
    lse_list = lse_flat.unbind(0)
    
    dycp_attn_output, _ = attention_lse_update_triton(list(out_list), list(lse_list))
    
    dycp_attn_output = dycp_attn_output.view(S, H, D)
    attn_output = attn_output.clone()
    attn_output[:num_dycp_reqs] = dycp_attn_output.to(attn_output.dtype)
    
    return attn_output
