"""Registered generalized MTP operator Meta dispatch contract."""

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def tensor(*shape, dtype=torch.int32):
    return torch.empty(shape, dtype=dtype, device="meta")


def test_generalized_lim_meta_accepts_packed_queries_and_state_outputs():
    batch, tokens, heads, blocks = 3, 11, 32, 128
    inputs = [
        tensor(tokens, heads, dtype=torch.bfloat16),
        tensor(tokens, heads, dtype=torch.float32),
        tensor(tokens, heads, 128, dtype=torch.bfloat16),
        tensor(blocks, 128, 1, dtype=torch.float32),
        tensor(blocks, 128, 1, 128, dtype=torch.bfloat16),
        tensor(batch, blocks),
        *[tensor(batch) for _ in range(6)],
        tensor(7, blocks * 128),
        tensor(tokens, 1, 2048),
        tensor(tokens, 1, 2048),
        tensor(tokens),
        tensor(batch, 32768),
        tensor(batch, 32768),
        tensor(batch),
    ]
    op = torch.ops._C_ascend.npu_fused_li_manage_mtp.default
    assert op(*inputs) is None
    writes = {arg.name for arg in op._schema.arguments if arg.alias_info is not None and arg.alias_info.is_write}
    assert writes == {
        "cache_slots_pool",
        "topk_src_ids",
        "topk_dst_slots",
        "topk_miss_counts",
        "miss_src_ids",
        "miss_dst_slots",
        "miss_counts",
    }
