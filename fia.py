"""
Single operator test for torch_npu.npu_fused_infer_attention_score
This file demonstrates how to call the fused attention operator similar to _forward_decode_pcp_dcp in attention_cp.py
"""

import torch
import torch_npu


def create_test_attention_operator():
    """Create test tensors and call npu_fused_infer_attention_score"""

    # User specified configuration
    # query: [4, 16, 128] -> [num_tokens, num_heads, head_size]
    num_tokens = 4
    num_heads = 16
    head_size = 128

    # k_nope: [268, 128, 128]
    # value: [268, 128, 128] same as k_nope
    num_kv_heads = 1

    # block_table: [4, 128], block_table[:,0]=1, others=0
    batch_size = 4
    max_blocks_per_seq = 128

    # actual_seq_lengths_kv: [3, 4, 4, 5]
    # actual_seq_lengths: [1, 2, 3, 4]
    actual_seq_lengths_kv = torch.tensor([3, 4, 4, 5], dtype=torch.int32, device='npu')
    actual_seq_lengths_q = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device='npu')

    # Create query tensor: [4, 16, 128] - layout TND
    query = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device='npu')

    # Create k_nope and value tensors: [268, 128, 128]
    # Based on the shape, this appears to be [batch*seq_len, num_kv_heads, head_size]
    # But num_kv_heads=1 so we need to interpret it differently
    # The shape might be [batch, seq_len, num_kv_heads * head_size] = [4, 67, 128] roughly
    # But user said [268, 128, 128], let's assume it's flat like [total_kv_tokens, head_dim]
    # Actually, let's try: treat as [batch, seq_len, head_dim] = [4, 268, 128] then view

    # For torch_npu.npu_fused_infer_attention_score:
    # - query: [num_tokens, num_heads, head_size] = TND layout
    # - k_nope: [batch, seq_len, num_kv_heads * head_size] or similar
    # - value: same as k_nope

    # Let's create based on user's shape interpretation
    # Assuming k_nope/value are [batch, seq_len, num_kv_heads * head_size]
    # But user gave [268, 128, 128] - let's try [batch, seq_len, head_dim] view
    k_nope = torch.randn(268, 128, 128, dtype=torch.float16, device='npu')
    value = torch.randn(268, 128, 128, dtype=torch.float16, device='npu')

    # Create block table: [4, 128], block_table[:,0]=1, others=0
    block_tables = torch.zeros((batch_size, max_blocks_per_seq), dtype=torch.int32, device='npu')
    block_tables[:, 0] = 1  # First block is valid

    # Scale for attention
    scale = 1.0 / (head_size ** 0.5)

    # Common kwargs similar to _forward_decode_pcp_dcp
    # sparse_mode not specified (will use default)
    common_kwargs = {
        "num_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "input_layout": "TND",
        "atten_mask": None,
        "scale": scale,
        "antiquant_mode": 0,
        "antiquant_scale": None,
        # "sparse_mode": 3,  # Not specified by user
        "softmax_lse_flag": True,
        "block_table": block_tables,
        "block_size": 16,  # Assuming block_size=16 based on typical config
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
    num_tokens = 4
    num_heads = 16
    head_size = 128
    num_kv_heads = 1
    batch_size = 4
    max_blocks_per_seq = 128

    # Create tensors
    query = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device='npu')
    k_nope = torch.randn(268, 128, 128, dtype=torch.float16, device='npu')
    value = torch.randn(268, 128, 128, dtype=torch.float16, device='npu')

    # block_table: [4, 128], block_table[:,0]=1, others=0
    block_tables = torch.zeros((batch_size, max_blocks_per_seq), dtype=torch.int32, device='npu')
    block_tables[:, 0] = 1

    actual_seq_lengths_kv = torch.tensor([3, 4, 4, 5], dtype=torch.int32, device='npu')
    actual_seq_lengths_q = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device='npu')
    scale = 1.0 / (head_size ** 0.5)

    common_kwargs = {
        "num_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "input_layout": "TND",
        "atten_mask": None,
        "scale": scale,
        "antiquant_mode": 0,
        "antiquant_scale": None,
        # "sparse_mode": 3,
        "softmax_lse_flag": True,
        "block_table": block_tables,
        "block_size": 16,
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