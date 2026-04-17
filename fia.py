"""
Single operator test for torch_npu.npu_fused_infer_attention_score
This file demonstrates how to call the fused attention operator similar to _forward_decode_pcp_dcp in attention_cp.py
"""

import torch
import torch_npu


def create_test_attention_operator():
    """Create test tensors and call npu_fused_infer_attention_score"""

    # Configuration similar to attention_cp.py
    batch_size = 2
    num_heads = 8
    num_kv_heads = 2
    head_size = 128
    num_tokens = 4  # number of query tokens
    max_seq_len = 2048
    block_size = 16

    # Create query tensor: [num_tokens, num_heads, head_size] - layout TND
    query = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device='npu')

    # Create key cache: [batch, block_num, block_size, num_kv_heads, head_size]
    num_blocks = max_seq_len // block_size
    key_cache = torch.randn(batch_size, num_blocks, block_size, num_kv_heads, head_size,
                           dtype=torch.float16, device='npu')
    value_cache = torch.randn(batch_size, num_blocks, block_size, num_kv_heads, head_size,
                             dtype=torch.float16, device='npu')

    # View key and value to get k_nope and value tensors
    # Shape: [batch, block_num * block_size, num_kv_heads * head_size]
    k_nope = key_cache.view(key_cache.shape[0], key_cache.shape[1] * key_cache.shape[2], -1)
    value = value_cache.view(value_cache.shape[0], value_cache.shape[1] * value_cache.shape[2], -1)

    # Create block table: [batch, max_blocks_per_seq]
    max_blocks_per_seq = 32
    block_tables = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq), dtype=torch.int32, device='npu')

    # Actual sequence lengths for KV (how many tokens are computed)
    actual_seq_lengths_kv = torch.tensor([100, 150], dtype=torch.int32, device='npu')

    # Actual sequence lengths for Q
    actual_seq_lengths_q = torch.tensor([num_tokens, num_tokens], dtype=torch.int32, device='npu')

    # Scale for attention
    scale = 1.0 / (head_size ** 0.5)

    # Common kwargs similar to _forward_decode_pcp_dcp
    common_kwargs = {
        "num_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "input_layout": "TND",
        "atten_mask": None,  # Could be a boolean mask if needed
        "scale": scale,
        "antiquant_mode": 0,
        "antiquant_scale": None,
        "sparse_mode": 3,
        "softmax_lse_flag": True,
        "block_table": block_tables,
        "block_size": block_size,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "actual_seq_lengths": actual_seq_lengths_q,
    }

    # Call the fused attention operator
    attn_out, attn_lse = torch_npu.npu_fused_infer_attention_score(
        query, k_nope, value, **common_kwargs
    )

    print(f"Query shape: {query.shape}")
    print(f"K_nope shape: {k_nope.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Attention output shape: {attn_out.shape}")
    print(f"Attention LSE shape: {attn_lse.shape}")

    return attn_out, attn_lse


def test_with_workspace():
    """
    Test with workspace - this is similar to the graph capturing path in attention_cp.py
    For graph capturing, we need to pre-allocate workspace and output buffers
    """
    batch_size = 2
    num_heads = 8
    num_kv_heads = 2
    head_size = 128
    num_tokens = 4
    max_seq_len = 2048
    block_size = 16
    num_blocks = max_seq_len // block_size

    # Create tensors
    query = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device='npu')
    key_cache = torch.randn(batch_size, num_blocks, block_size, num_kv_heads, head_size,
                           dtype=torch.float16, device='npu')
    value_cache = torch.randn(batch_size, num_blocks, block_size, num_kv_heads, head_size,
                             dtype=torch.float16, device='npu')

    k_nope = key_cache.view(key_cache.shape[0], key_cache.shape[1] * key_cache.shape[2], -1)
    value = value_cache.view(value_cache.shape[0], value_cache.shape[1] * value_cache.shape[2], -1)

    block_tables = torch.randint(0, num_blocks, (batch_size, 32), dtype=torch.int32, device='npu')
    actual_seq_lengths_kv = torch.tensor([100, 150], dtype=torch.int32, device='npu')
    actual_seq_lengths_q = torch.tensor([num_tokens, num_tokens], dtype=torch.int32, device='npu')
    scale = 1.0 / (head_size ** 0.5)

    common_kwargs = {
        "num_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "input_layout": "TND",
        "atten_mask": None,
        "scale": scale,
        "antiquant_mode": 0,
        "antiquant_scale": None,
        "sparse_mode": 3,
        "softmax_lse_flag": True,
        "block_table": block_tables,
        "block_size": block_size,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "actual_seq_lengths": actual_seq_lengths_q,
    }

    # Get workspace size first
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        query, k_nope, value, **common_kwargs
    )

    # Pre-allocate output buffers
    attn_out = torch.empty_like(query)
    attn_lse = torch.empty((num_tokens, num_heads, 1), dtype=torch.float, device='npu')

    # Call with workspace and output buffers
    torch_npu.npu_fused_infer_attention_score.out(
        query, k_nope, value, **common_kwargs,
        workspace=workspace,
        out=[attn_out, attn_lse]
    )

    print(f"Test with workspace - Output shape: {attn_out.shape}, LSE shape: {attn_lse.shape}")
    return attn_out, attn_lse


if __name__ == "__main__":
    print("=" * 50)
    print("Test 1: Basic call without workspace")
    print("=" * 50)
    create_test_attention_operator()

    print("\n" + "=" * 50)
    print("Test 2: Call with workspace (graph path)")
    print("=" * 50)
    test_with_workspace()