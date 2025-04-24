# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/kernels/test_moe.py
# Copyright 2023 The vLLM team.
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
# import torch_npu
# import math
from typing import Optional, Tuple, Union, List

def concat_and_cache_mla_torch(
    kv_c_normed: torch.Tensor,  # [num_tokens, num_kv_head, nope]
    k_pe: torch.Tensor,         # [num_tokens, num_kv_head, rope]
    kv_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_head, nope + rope]
    slot_mapping,               # [num_tokens]
):
  num_blocks = kv_cache.size()[0]
  block_size = kv_cache.size()[1]
  num_kv_head = k_pe.size()[1]

  idx_for_copy = slot_mapping // block_size * block_size + slot_mapping % block_size
  kv_cache = kv_cache.view(num_blocks * block_size, num_kv_head, -1)
  kv_cache[idx_for_copy] = torch.cat([kv_c_normed.unsqueeze(1), k_pe], dim=-1)


# forked https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_cascade_flash_attn.py#L56-L64
def merge_attn_states_torch(
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
        output_lse: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Reference implementation.
    dtype = prefix_output.dtype
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    p_lse = torch.exp(prefix_lse - max_lse)
    s_lse = torch.exp(suffix_lse - max_lse)
    p_scale = p_lse / (p_lse + s_lse)
    s_scale = s_lse / (p_lse + s_lse)
    p_scale = p_scale.transpose(0, 1).unsqueeze(2)
    s_scale = s_scale.transpose(0, 1).unsqueeze(2)
    output = p_scale * prefix_output + s_scale * suffix_output
    output = output.to(dtype)
    return output

# forked https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_cache.py#L726-L742
def gather_cache_torch(
        src_cache: torch.Tensor,     # [NUM_BLOCKS, BLOCK_SIZE, HEAD, ENTRIES]
        dst: torch.Tensor,           # [TOT_TOKENS, ENTRIES]
        block_table: torch.Tensor,   # [BATCH, BLOCK_INDICES]
        cu_seq_lens: torch.Tensor,   # [BATCH+1]
        batch_size: int,
        seq_starts: Optional[torch.Tensor] = None  # Optional: [BATCH]
   ) -> None:
        """
        从源缓存中收集序列数据到目标tensor
        Args:
        src_cache: 源缓存tensor [NUM_BLOCKS, BLOCK_SIZE, HEAD, ENTRIES]
        dst: 目标tensor [TOT_TOKENS, ENTRIES]
        block_table: 块表映射 [BATCH, BLOCK_INDICES]
        cu_seq_lens: 累积序列长度 [BATCH+1]
        batch_size: 批大小
        seq_starts: 可选,每个batch的起始偏移 [BATCH]
        """
        # 基本参数检查
        assert src_cache.dtype == dst.dtype, "src_cache and dst must have same dtype"
        assert block_table.dtype == torch.int32, "block_table must be int32"
        assert cu_seq_lens.dtype == torch.int32, "cu_seq_lens must be int32"
        
        if seq_starts is not None:
            assert seq_starts.dtype == torch.int32, "seq_starts must be int32"
            
        block_size = src_cache.size(1)
        # 对每个batch进行处理
        for bid in range(batch_size):
            # 获取当前batch的序列起始和结束位置
            seq_start = cu_seq_lens[bid].item()
            seq_end = cu_seq_lens[bid + 1].item()
            seq_len = seq_end - seq_start
            
            if seq_len == 0:
                continue
            
            # 计算需要的block数
            tot_blocks = (seq_len + block_size - 1) // block_size
            
            # 如果有seq_starts,计算block偏移
            offset = 0
            if seq_starts is not None:
                offset = seq_starts[bid].item() // block_size

            # 获取当前batch的block table
            batch_block_table = block_table[bid, offset:offset + tot_blocks]
            # 计算完整blocks和最后一个partial block
            full_blocks = tot_blocks - 1 if seq_len % block_size else tot_blocks
            partial_block_size = seq_len % block_size if seq_len % block_size else 0
            # 复制完整blocks
            dst_start = seq_start
            for i in range(full_blocks):
                block_id = batch_block_table[i].item()
                # 复制整个block，移除HEAD维度
                dst[dst_start:dst_start + block_size] = src_cache[block_id].squeeze(1)
                dst_start += block_size
                
            # 处理最后一个不完整block
            if partial_block_size > 0:
                block_id = batch_block_table[full_blocks].item()
                dst[dst_start:dst_start + partial_block_size] = \
                src_cache[block_id, :partial_block_size].squeeze(1)



#forked https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flashmla.py#L86-L104
def flash_attn_varlen_func_torch(
    q: torch.Tensor,                # [total_tokens, num_heads, head_size]
    k: torch.Tensor,                # [total_tokens, num_heads, head_size]
    v: torch.Tensor,                # [total_tokens, num_heads, head_size]
    cu_seqlens_q: torch.Tensor,     # [batch_size + 1], cumulative sequence lengths for query
    cu_seqlens_k: torch.Tensor,     # [batch_size + 1], cumulative sequence lengths for key/value
    max_seqlen_q: int,             # maximum query sequence length
    max_seqlen_k: int,             # maximum key/value sequence length
    softmax_scale: float,          # scale factor for attention scores
    causal: bool = True,           # whether to apply causal mask
    return_softmax_lse: bool = False  # whether to return log sum exp of softmax
) -> Union[torch.Tensor, torch.Tensor]:
    """Reference implementation of flash attention with variable length sequences.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        cu_seqlens_q: Cumulative query sequence lengths
        cu_seqlens_k: Cumulative key sequence lengths
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        softmax_scale: Scale factor for attention scores
        causal: Whether to apply causal mask
        return_softmax_lse: Whether to return log sum exp of softmax
    
    Returns:
        output: Attention output
        lse: Optional log sum exp of softmax if return_softmax_lse is True
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    device = q.device
    dtype = q.dtype
    
    ref_output: List[torch.Tensor] = []
    lse_list: List[torch.Tensor] = []
    
    for i in range(batch_size):
        # Get sequence lengths for current batch
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        k_start = cu_seqlens_k[i].item()
        k_end = cu_seqlens_k[i + 1].item()
        
        q_len = q_end - q_start
        k_len = k_end - k_start
        
        if q_len == 0 or k_len == 0:
            continue
            
        # Get current batch tensors
        curr_q = q[q_start:q_end]  # [q_len, num_heads, head_size]
        curr_k = k[k_start:k_end]  # [k_len, num_heads, head_size]
        curr_v = v[k_start:k_end]  # [k_len, num_heads, head_size]
        
        # Apply scale
        curr_q = curr_q * softmax_scale
        
        # Calculate attention scores
        attn = torch.einsum("qhd,khd->hqk", curr_q, curr_k).float()
        
        # Apply causal mask if needed
        if causal:
            mask = torch.triu(
                torch.ones(q_len, k_len, device=device), 
                diagonal=k_len - q_len + 1
            ).bool()
            attn.masked_fill_(mask, float("-inf"))
            
        # Calculate softmax and output
        attn_softmax = torch.softmax(attn, dim=-1).to(dtype)
        out = torch.einsum("hqk,khd->qhd", attn_softmax, curr_v)
        
        # Store outputs
        ref_output.append(out)
        
        # Calculate log sum exp if needed
        if return_softmax_lse:
            lse = torch.logsumexp(attn, dim=-1)  # [num_heads, q_len]
            lse_list.append(lse)
    
    output = torch.cat(ref_output, dim=0) if ref_output else \
             torch.empty(0, q.size(1), q.size(2), device=device, dtype=dtype)
             
    if return_softmax_lse:
        lse = torch.cat(lse_list, dim=1) if lse_list else \
                              torch.empty(q.size(1), 0, device=device, dtype=torch.float32)
        return output, lse
        
    return output


# forked https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flashmla.py#L86-L120
# forked https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla.py#L84-L100
def flash_mla_with_kvcache_torch(
    q: torch.Tensor,                      # [b, s_q=1, h_q, d]
    k_cache: torch.Tensor,                # [num_blocks, block_size, h_kv=1, d]
    block_table: torch.Tensor,            # [b, max_seqlen//block_size]
    cache_seqlens: torch.Tensor,          # [b]
    head_dim_v: int,                      # value dimension (kv_lora_rank)
    softmax_scale: float = 1.0,           # attention scale factor
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for flash_mla_with_kvcache.
    
    Args:
        q: Query tensor with shape [batch, 1, num_heads_q, head_dim]
        k_cache: Key cache tensor with shape [num_blocks, block_size, 1, head_dim]
        block_table: Block mapping table [batch, max_seqlen//block_size]
        cache_seqlens: Sequence lengths [batch]
        head_dim_v: Value head dimension (kv_lora_rank)
        softmax_scale: Scale factor for attention scores
        causal: Whether to use causal attention
    
    Returns:
        output: Output tensor [batch, 1, num_heads_q, head_dim_v]
        lse: Log sum exp [batch, num_heads_q, 1]
    """
    b, s_q, h_q, d = q.shape
    assert s_q == 1, "Decoding only supports sequence length 1"
    
    num_blocks, block_size, h_kv, _ = k_cache.shape
    assert h_kv == 1, "Key cache should have 1 head dimension"
    
    device = q.device
    dtype = q.dtype
    
    # Initialize output tensors
    out = torch.empty(b, 1, h_q, head_dim_v, dtype=dtype, device=device)
    lse = torch.empty(b, h_q, 1, dtype=torch.float32, device=device)
    
    # Scale query
    q = q * softmax_scale
    
    for i in range(b):
        # Get sequence length for this batch
        seq_len = cache_seqlens[i].item()
        
        # Get query for this batch
        q_i = q[i]  # [1, h_q, d]
        
        # Get block indices for this batch
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_indices = block_table[i, :num_blocks_needed]
        
        # Gather keys for this batch using block indices
        k_i = k_cache[block_indices]  # [num_blocks_needed, block_size, 1, d]
        k_i = k_i.view(-1, h_kv, d)[:seq_len]  # [seq_len, 1, d]
        
        # Repeat k for multi-query attention
        k_i = k_i.repeat_interleave(h_q, dim=1)  # [seq_len, h_q, d]
        
        # Compute attention scores
        scores = torch.einsum('nhd,khd->nhk', q_i, k_i)  # [1, h_q, seq_len]
        
        # Apply causal mask if needed
        if causal:
            scores.masked_fill_(
                torch.arange(seq_len, device=device) > seq_len - 1,
                float('-inf')
            )
        
        # Compute log sum exp
        lse[i] = scores.logsumexp(dim=-1, keepdim=True)  # [1, h_q, 1]
        
        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # [1, h_q, seq_len]
        
        # Compute output
        # Use head_dim_v instead of full dimension for values
        v_i = k_i[..., :head_dim_v]  # [seq_len, h_q, head_dim_v]
        out[i] = torch.einsum('nhk,khd->nhd', attn_weights, v_i)  # [1, h_q, head_dim_v]
    
    return out, lse
