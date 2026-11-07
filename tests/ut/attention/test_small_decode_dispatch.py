# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm_ascend.attention.attention_v1 as attention


@pytest.mark.parametrize(
    "enabled,max_model_len,num_prefills,max_query_len,use_vector",
    [
        (True, 1024, 0, 1, True),
        (False, 1024, 0, 1, False),
        (True, 1025, 0, 1, False),
        (True, 1024, 1, 1, False),
        (True, 1024, 0, 2, False),
    ],
)
def test_capture_dispatch_preserves_fia_fallback(enabled, max_model_len, num_prefills, max_query_len, use_vector):
    impl = attention.AscendAttentionBackendImpl.__new__(attention.AscendAttentionBackendImpl)
    impl.num_kv_heads = 8
    impl.sinks = impl.alibi_slopes = impl.sliding_window = None
    impl.attn_type = attention.AttentionType.DECODER
    impl.use_bnsd_kv_cache = impl.enable_c8_quant = False
    impl.kv_sharing_target_layer_name = None
    impl.scale = 0.125
    impl.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        speculative_config=None,
        kv_transfer_config=None,
    )
    backing = torch.empty(2, 2, 128, 8, 128, dtype=torch.bfloat16)
    impl.key_cache, impl.value_cache = backing[:, 0], backing[:, 1]
    query = torch.empty(1, 16, 128, dtype=torch.bfloat16)
    output = torch.zeros(1, 16 * 128, dtype=query.dtype)
    metadata = SimpleNamespace(
        max_query_len=max_query_len,
        num_prefills=num_prefills,
        causal=True,
        seq_lens_gpu=torch.tensor([129], dtype=torch.int32),
        block_tables=torch.tensor([[1, 0]], dtype=torch.int32),
        uses_pcp_sharded_layout=False,
    )
    impl.full_graph_fia = Mock(return_value=(output, 1))
    with (
        patch.object(
            attention, "get_ascend_config", return_value=SimpleNamespace(enable_small_decode_attention=enabled)
        ),
        patch.object(attention.envs_vllm, "VLLM_BATCH_INVARIANT", False),
        patch.object(attention, "_EXTRA_CTX", SimpleNamespace(capturing=True)),
        patch.object(attention, "small_decode_attention", side_effect=lambda *args: args[-1].fill_(42)) as operator,
    ):
        result = impl.forward_fused_infer_attention(query, query, query, metadata, output)
    assert result is output
    if use_vector:
        operator.assert_called_once()
        assert operator.call_args.args[1] is impl.key_cache
        assert operator.call_args.args[3] is metadata.block_tables
        assert operator.call_args.args[4] is metadata.seq_lens_gpu
        assert torch.all(output == 42)
        impl.full_graph_fia.assert_not_called()
    else:
        operator.assert_not_called()
        impl.full_graph_fia.assert_called_once()
