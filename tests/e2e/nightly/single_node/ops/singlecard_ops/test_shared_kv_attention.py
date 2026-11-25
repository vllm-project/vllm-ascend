# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.device.device_config import is_310p, is_950


@pytest.mark.skipif(is_310p() or is_950(), reason="Gemma4 large-head prefill targets A2/A3")
@pytest.mark.parametrize("head_size,sliding_window", [(256, 128), (512, None)])
@pytest.mark.parametrize("cached_prefix", [False, True])
@torch.inference_mode()
def test_shared_kv_attention_matches_dense_reference(head_size, sliding_window, cached_prefix):
    generator = torch.Generator().manual_seed(42)
    num_heads = 4
    seq_lens = [131, 145]
    query_lens = [3, 5] if cached_prefix else seq_lens
    num_tokens = sum(query_lens)
    query = torch.randn(num_tokens, num_heads, head_size, generator=generator, dtype=torch.bfloat16)
    key_cache = torch.randn(5, 128, 1, head_size, generator=generator, dtype=torch.bfloat16)
    value_cache = torch.randn(5, 128, 1, head_size, generator=generator, dtype=torch.bfloat16)
    block_tables = torch.tensor([[3, 0], [1, 4]], dtype=torch.int32)
    expected = []
    offset = 0
    for row, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        key = key_cache[block_tables[row].long()].reshape(-1, 1, head_size)[:seq_len]
        value = value_cache[block_tables[row].long()].reshape(-1, 1, head_size)[:seq_len]
        query_positions = torch.arange(seq_len - query_len, seq_len).unsqueeze(1)
        key_positions = torch.arange(seq_len).unsqueeze(0)
        visible = key_positions <= query_positions
        if sliding_window is not None:
            visible &= key_positions >= query_positions - sliding_window
        expected.append(
            F.scaled_dot_product_attention(
                query[offset : offset + query_len].transpose(0, 1).float(),
                key.transpose(0, 1).float().expand(num_heads, -1, -1),
                value.transpose(0, 1).float().expand(num_heads, -1, -1),
                attn_mask=visible,
            ).transpose(0, 1)
        )
        offset += query_len

    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.kv_sharing_target_layer_name = "producer"
    impl.key_cache = key_cache.npu()
    impl.value_cache = value_cache.npu()
    impl.use_bnsd_kv_cache = False
    impl.sinks = None
    impl.sliding_window = sliding_window
    impl.num_heads = num_heads
    impl.num_kv_heads = 1
    impl.head_size = head_size
    impl.scale = head_size**-0.5
    metadata = AscendMetadata(
        attn_state=AscendAttentionState.PrefillCacheHit if cached_prefix else AscendAttentionState.PrefillNoCache,
        num_actual_tokens=num_tokens,
        actual_seq_lengths_q=[query_lens[0], num_tokens],
        seq_lens_list=seq_lens,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32).npu(),
        block_tables=block_tables.npu(),
        max_query_len=max(query_lens),
        attn_mask=AttentionMaskBuilder(torch.device("npu")).get_splitfuse_attn_mask(),
    )
    query = query.npu()
    with patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(capturing=False)):
        output = impl.forward_fused_infer_attention(query, None, None, metadata, torch.empty_like(query))
    torch.testing.assert_close(output.cpu().float(), torch.cat(expected), atol=1e-2, rtol=1e-2)
