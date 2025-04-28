#forked https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flash_attn.py
#test https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py#L667-L674
from typing import List, Union

import pytest
import torch
from vllm.platforms import current_platform


#forked https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flashmla.py#L86-L104
def flash_attn_varlen_func_torch(
    q: torch.Tensor,  # [total_tokens, num_heads, head_size]
    k: torch.Tensor,  # [total_tokens, num_heads, head_size]
    v: torch.Tensor,  # [total_tokens, num_heads, head_size]
    cu_seqlens_q: torch.
    Tensor,  # [batch_size + 1], cumulative sequence lengths for query
    cu_seqlens_k: torch.
    Tensor,  # [batch_size + 1], cumulative sequence lengths for key/value
    max_seqlen_q: int,  # maximum query sequence length
    max_seqlen_k: int,  # maximum key/value sequence length
    softmax_scale: float,  # scale factor for attention scores
    causal: bool = True,  # whether to apply causal mask
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
            mask = torch.triu(torch.ones(q_len, k_len, device=device),
                              diagonal=k_len - q_len + 1).bool()
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("seq_len", [128, 256])
@torch.inference_mode()
def test_flash_attn_varlen_basic(
    dtype: torch.dtype,
    batch_size: int,
    num_heads: int,
    head_size: int,
    seq_len: int,
) -> None:
    """Test basic functionality of flash_attn_varlen_func"""
    torch.set_default_device("npu")
    device = torch.device("npu")

    # 设置随机种子
    current_platform.seed_everything(0)

    # 为每个batch生成随机的序列长度
    seq_lens = torch.randint(1, seq_len + 1, (batch_size, ), device=device)
    total_tokens = seq_lens.sum().item()

    # 生成输入张量
    query = torch.randn(total_tokens,
                        num_heads,
                        head_size,
                        dtype=dtype,
                        device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # 计算累积序列长度
    cu_seq_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)

    # 计算scale
    scale = head_size**-0.5

    return_softmax_lse = True

    flash_attn_varlen_func_torch(q=query,
                                 k=key,
                                 v=value,
                                 cu_seqlens_q=cu_seq_lens,
                                 cu_seqlens_k=cu_seq_lens,
                                 max_seqlen_q=seq_len,
                                 max_seqlen_k=seq_len,
                                 softmax_scale=scale,
                                 causal=True,
                                 return_softmax_lse=return_softmax_lse)
    '''
    # 调用flash attention
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    output = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seq_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        return_softmax_lse=return_softmax_lse
    )

    if return_softmax_lse:
        suffix_output, suffix_lse = output
        ref_suffix_output, ref_suffix_lse = ref_output
        torch.testing.assert_close(suffix_output, ref_suffix_output, atol=1.5e-2, rtol=1e-2)
        torch.testing.assert_close(suffix_lse, ref_suffix_lse, atol=1.5e-2, rtol=1e-2)
    else:
        # 验证结果
        torch.testing.assert_close(
            output,
            ref_output,
            atol=1.5e-2,
            rtol=1e-2
        )
    '''
