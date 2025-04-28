#forked https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flashmla.py
#test https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/flashmla.py#L136-L146
import random
from typing import Tuple

import pytest
import torch


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    #RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max(
        (x * x + y * y).sum().item(), 1e-12)
    #amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    assert cos_diff < 1e-5


def flash_mla_with_kvcache_torch(
    q: torch.Tensor,  # [b, s_q=1, h_q, d]
    k_cache: torch.Tensor,  # [num_blocks, block_size, h_kv=1, d]
    block_table: torch.Tensor,  # [b, max_seqlen//block_size]
    cache_seqlens: torch.Tensor,  # [b]
    head_dim_v: int,  # value dimension (kv_lora_rank)
    softmax_scale: float = 1.0,  # attention scale factor
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
                float('-inf'))

        # Compute log sum exp
        lse[i] = scores.logsumexp(dim=-1, keepdim=True)  # [1, h_q, 1]

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # [1, h_q, seq_len]

        # Compute output
        # Use head_dim_v instead of full dimension for values
        v_i = k_i[..., :head_dim_v]  # [seq_len, h_q, head_dim_v]
        out[i] = torch.einsum('nhk,khd->nhd', attn_weights,
                              v_i)  # [1, h_q, head_dim_v]

    return out, lse


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_len_q", [1])  # MTP = 1, 2
@pytest.mark.parametrize("mean_seq_len_k", [4096, 8192])
@pytest.mark.parametrize("num_heads_q", [16, 32, 64, 128])  # TP = 8, 4, 2, 1
@pytest.mark.parametrize("head_dim", [576])
@pytest.mark.parametrize("head_dim_v", [512])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_flash_mla_with_kvcache(
    batch_size: int,
    seq_len_q: int,
    mean_seq_len_k: int,
    num_heads_q: int,
    head_dim: int,
    head_dim_v: int,
    block_size: int,
    varlen: bool,
    dtype: torch.dtype,
) -> None:
    """Test flash_mla_with_kvcache against reference PyTorch implementation."""
    device = torch.device("npu")
    torch.manual_seed(0)
    random.seed(0)

    # 固定参数
    num_heads_kv = 1
    causal = True

    # 设置序列长度
    cache_seqlens = torch.full((batch_size, ),
                               mean_seq_len_k,
                               dtype=torch.int32,
                               device=device)
    if varlen:
        for i in range(batch_size):
            cache_seqlens[i] = max(
                int(random.normalvariate(mean_seq_len_k, mean_seq_len_k / 2)),
                seq_len_q)

    # 计算最大序列长度和padding
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = ((max_seqlen + 255) // 256) * 256

    # 创建输入张量
    q = torch.randn(batch_size,
                    seq_len_q,
                    num_heads_q,
                    head_dim,
                    device=device,
                    dtype=dtype)

    # 创建block table
    num_blocks = batch_size * max_seqlen_pad // block_size
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)
    block_table = block_table.view(batch_size, max_seqlen_pad // block_size)

    # 创建key cache
    k_cache = torch.randn(num_blocks,
                          block_size,
                          num_heads_kv,
                          head_dim,
                          device=device,
                          dtype=dtype)

    # 设置padding区域为NaN
    for i in range(batch_size):
        k_cache.view(batch_size, max_seqlen_pad, num_heads_kv,
                     head_dim)[i, cache_seqlens[i].item():] = float('nan')

    # 计算scale
    scale = head_dim**-0.5

    # PyTorch参考实现
    out_torch, lse_torch = flash_mla_with_kvcache_torch(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        softmax_scale=scale,
        causal=causal,
    )
    '''
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    # 获取tile scheduler metadata
    tile_metadata, num_splits = get_mla_metadata(
        cache_seqlens, 
        seq_len_q * num_heads_q // num_heads_kv,
        num_heads_kv
    )

    # Flash MLA实现
    out_flash, lse_flash = flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_metadata,
        num_splits=num_splits,
        softmax_scale=scale,
        causal=causal,
    )
    
    # 验证结果
    cal_diff(out_flash, out_torch, "output")
    cal_diff(lse_flash, lse_torch, "lse")
    '''
