"""Registered generalized MTP operator Meta dispatch contract."""

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def tensor(*shape, dtype=torch.int32):
    return torch.empty(shape, dtype=dtype, device="meta")


def test_generalized_copy_meta_accepts_per_query_and_per_request_misses():
    batch, tokens, heads, blocks = 3, 22, 8, 128
    inputs = [
        tensor(tokens, heads, 64, dtype=torch.bfloat16),
        tensor(tokens, heads, 512, dtype=torch.bfloat16),
        tensor(batch),
        tensor(batch),
        tensor(batch),
        tensor(tokens, 1, 2048),
        tensor(tokens, 1, 2048),
        tensor(tokens),
        tensor(batch, 32768),
        tensor(batch, 32768),
        tensor(batch),
        tensor(batch, blocks),
        tensor(batch, blocks),
        tensor(blocks, 128, 1, 64, dtype=torch.bfloat16),
        tensor(blocks, 128, 1, 512, dtype=torch.bfloat16),
        tensor(blocks, 128, 64, dtype=torch.bfloat16),
        tensor(blocks, 128, 512, dtype=torch.bfloat16),
        0.0442,
        tensor(tokens, heads, 512, dtype=torch.bfloat16),
    ]
    op = torch.ops._C_ascend.npu_fused_copy_sfa_mtp.default
    assert op(*inputs) is None
    writes = {arg.name for arg in op._schema.arguments if arg.alias_info is not None and arg.alias_info.is_write}
    assert writes == {"hbm_k_rope", "hbm_kv_cache", "attention_out"}
