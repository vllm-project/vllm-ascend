# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved

# Third Party
import torch

KVCACHE_NZ_DIM = 16

def check_paged_kv_cache_equal_with_tuple(
    left, right, num_tokens, slot_mapping, lora, rope, block_size, num_blocks
):
    for left_kv, right_kv in zip(left, right, strict=False):
        lora_left = left_kv[0].cpu()
        rope_left = left_kv[1].cpu()
        lora_right = right_kv[0].cpu()
        rope_right = right_kv[1].cpu()
        
        lora_left = lora_left.view(num_blocks, 1, lora // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM).transpose(1, 3).reshape(-1, lora).index_select(0, slot_mapping).contiguous()
        rope_left = rope_left.view(num_blocks, 1, rope // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM).transpose(1, 3).reshape(-1, rope).index_select(0, slot_mapping).contiguous()
        lora_right = lora_right.view(num_blocks, 1, lora // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM).transpose(1, 3).reshape(-1, lora).index_select(0, slot_mapping).contiguous()
        rope_right = rope_right.view(num_blocks, 1, rope // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM).transpose(1, 3).reshape(-1, rope).index_select(0, slot_mapping).contiguous()
        
        assert (lora_left == lora_right).all()
        assert (rope_left == rope_right).all()

def generate_kv_cache_paged_list_tensors(
    num_blocks,
    device,
    block_size,
    num_layers,
    num_heads,
    head_size,
    dtype=torch.bfloat16,
    use_mla=False
):
    """
    Instead of Tuple[Tuple[Tensor, Tensor]], return List[Tensor]
    where KV are in the same tensor
    """
    ret = []
    num_heads = 1 if use_mla else num_heads
    shape = (
        [num_blocks, block_size, head_size]
        if use_mla
        else [2, num_blocks, block_size, num_heads, head_size]
    )

    for i in range(num_layers):
        kv = torch.rand(shape, dtype=dtype, device=device)
        ret.append(kv)

    return ret

def generate_kv_cache_paged_list_tensors_tuple(
    num_blocks,
    device,
    block_size,
    num_layers,
    lora,
    rope,
    dtype=torch.bfloat16,
):
    """
    Instead of Tuple[Tuple[Tensor, Tensor]], return List[Tuple[Tensor, Tensor]]
    where KV are in the same tensor
    """
    ret = []
    num_heads = 1
    lora_shape = (
        [num_blocks, block_size, num_heads, lora]
    )
    rope_shape = (
        [num_blocks, block_size, num_heads, rope]
    )

    for i in range(num_layers):
        kv_lora = torch.rand(lora_shape, dtype=dtype, device=device)
        kv_rope = torch.rand(rope_shape, dtype=dtype, device=device)
        ret.append((kv_lora, kv_rope))
    return ret

def check_paged_kv_cache_equal(
    left, right, num_tokens, slot_mapping, num_heads=8, head_size=128
):
    """
    check whether two paged kv caches are the same at slot_mapping
    """
    token_dim = 0
    for left_kv, right_kv in zip(left, right, strict=False):
        left_k = left_kv[0].reshape(-1, num_heads, head_size)
        left_v = left_kv[1].reshape(-1, num_heads, head_size)
        right_k = right_kv[0].reshape(-1, num_heads, head_size)
        right_v = right_kv[1].reshape(-1, num_heads, head_size)

        assert len(left_k.shape) == 3
        assert len(left_v.shape) == 3
        assert len(right_k.shape) == 3
        assert len(right_v.shape) == 3

        assert left_k.shape[token_dim] >= num_tokens
        assert left_v.shape[token_dim] >= num_tokens
        assert right_k.shape[token_dim] >= num_tokens
        assert right_v.shape[token_dim] >= num_tokens

        assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
        assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()
