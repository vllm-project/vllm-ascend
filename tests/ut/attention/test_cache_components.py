# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.cache_components import bind_component
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend


def test_gqa_kernel_blocks_are_interleaved_within_manager_pages():
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.bfloat16)
    components = AscendAttentionBackend.describe_cache_components(spec, 2, 2, 32)
    raw = torch.zeros(80, dtype=torch.uint8)
    manager_view = raw[16:].view(2, 32)
    key, value = [bind_component(manager_view, c, 64) for c in components]
    key[2].fill_(2)
    value[3].fill_(3)
    expected = torch.zeros_like(raw)
    expected[48:56].view(torch.bfloat16).fill_(2)
    expected[72:80].view(torch.bfloat16).fill_(3)
    torch.testing.assert_close(raw, expected)


def test_gdn_mixed_dtype_components_preserve_page_padding():
    spec = MambaSpec(
        block_size=4,
        shapes=((2, 2), (1, 2, 2)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=32,
    )
    components = AscendGDNAttentionBackend.describe_cache_components(spec, 2, 4, 32)
    raw = torch.zeros(80, dtype=torch.uint8)
    manager_view = raw[16:].view(2, 32)
    conv, state = [bind_component(manager_view, c, 64) for c in components]
    assert conv.shape == (2, 2, 2)
    assert state.shape == (2, 1, 2, 2)
    conv[1].fill_(2)
    state[0].fill_(3)
    expected = torch.zeros_like(raw)
    expected[48:56].view(torch.bfloat16).fill_(2)
    expected[24:40].view(torch.float32).fill_(3)
    torch.testing.assert_close(raw, expected)
