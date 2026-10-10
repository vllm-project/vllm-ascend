# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3SparseAttention
from vllm_ascend.models.minimax_m3.msa_m3 import (
    minimax_m3_fused_sparse_forward,
    minimax_m3_fused_sparse_forward_fake,
)


def make_layer():
    norm = SimpleNamespace(variance_epsilon=1e-6, weight=torch.zeros(128))
    return SimpleNamespace(
        indexer_proj=None,
        head_dim=128,
        idx_head_dim=128,
        rotary_emb=SimpleNamespace(is_neox_style=True, rotary_dim=64),
        q_norm=norm,
        k_norm=norm,
        index_q_norm=norm,
        index_k_norm=norm,
    )


@pytest.mark.parametrize(
    "unsupported", [None, "dtype", "cpu", "positions", "heads", "rope", "eps", "triton", "large_batch"]
)
def test_fusion_guard(unsupported):
    layer = make_layer()
    hidden = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, shape=(4, 128))
    positions = torch.zeros(4, dtype=torch.int64)
    if unsupported == "dtype":
        hidden.dtype = torch.float16
    elif unsupported == "cpu":
        hidden.device.type = "cpu"
    elif unsupported == "positions":
        positions = positions.view(2, 2)
    elif unsupported == "heads":
        layer.idx_head_dim = 64
    elif unsupported == "rope":
        layer.rotary_emb.is_neox_style = False
    elif unsupported == "eps":
        layer.k_norm = SimpleNamespace(variance_epsilon=1e-5)
    elif unsupported == "large_batch":
        hidden.shape = (513, 128)
    with patch("vllm_ascend.models.minimax_m3.minimax_m3.HAS_TRITON", unsupported != "triton"):
        assert MiniMaxM3SparseAttention._can_fuse_sparse_prepare(layer, positions, hidden) == (unsupported is None)


def test_fused_forward_skips_materialized_prepare_and_insert():
    hidden = torch.randn(4, 128, dtype=torch.bfloat16)
    qkv = torch.randn(4, 1536, dtype=torch.bfloat16)
    positions = torch.arange(4)
    layer = SimpleNamespace(
        _can_fuse_sparse_prepare=Mock(return_value=True),
        qkv_proj=Mock(return_value=(qkv, None)),
        indexer_proj=None,
        _sparse_prepare=Mock(side_effect=AssertionError("separate prepare must not run")),
        q_size=1024,
        layer_name="layers.0.attn",
        o_proj=Mock(side_effect=lambda x: (x, None)),
    )
    with patch.object(torch.ops.vllm, "minimax_m3_fused_sparse_forward") as fused:
        output = MiniMaxM3SparseAttention.forward(layer, positions, hidden)
    layer.qkv_proj.assert_called_once_with(hidden)
    fused.assert_called_once_with(qkv, positions, output, layer.layer_name, None)
    assert output.shape == (4, 1024) and output.dtype == hidden.dtype
    layer._sparse_prepare.assert_not_called()


@pytest.mark.parametrize("tokens, supported", [(512, True), (513, False)])
def test_fusion_token_limit(tokens, supported):
    hidden = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, shape=(tokens, 128))
    with patch("vllm_ascend.models.minimax_m3.minimax_m3.HAS_TRITON", True):
        assert (
            MiniMaxM3SparseAttention._can_fuse_sparse_prepare(make_layer(), torch.arange(tokens), hidden) == supported
        )


def test_separate_index_projection_uses_fusion_without_concatenation():
    hidden = torch.randn(4, 128, dtype=torch.bfloat16)
    qkv = torch.randn(4, 1280, dtype=torch.bfloat16)
    index_qk = torch.randn(4, 256, dtype=torch.bfloat16)
    positions = torch.arange(4)
    layer = SimpleNamespace(
        _can_fuse_sparse_prepare=Mock(return_value=True),
        qkv_proj=Mock(return_value=(qkv, None)),
        indexer_proj=Mock(return_value=(index_qk, None)),
        q_size=1024,
        layer_name="layers.0.attn",
        o_proj=Mock(side_effect=lambda x: (x, None)),
    )
    with (
        patch.object(torch.ops.vllm, "minimax_m3_fused_sparse_forward") as fused,
        patch("torch.cat", side_effect=AssertionError("projections must not be concatenated")),
    ):
        output = MiniMaxM3SparseAttention.forward(layer, positions, hidden)
    layer.indexer_proj.assert_called_once_with(hidden)
    fused.assert_called_once_with(qkv, positions, output, layer.layer_name, index_qk)

    guard_layer = make_layer()
    guard_layer.indexer_proj = layer.indexer_proj
    hidden_meta = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, shape=(4, 128))
    with patch("vllm_ascend.models.minimax_m3.minimax_m3.HAS_TRITON", True):
        assert MiniMaxM3SparseAttention._can_fuse_sparse_prepare(guard_layer, positions, hidden_meta)


def test_fused_op_dummy_does_not_touch_cache():
    output = torch.ones(4, 1024)
    context = SimpleNamespace(attn_metadata=None, no_compile_layers={})
    with patch("vllm_ascend.models.minimax_m3.msa_m3.get_forward_context", return_value=context):
        minimax_m3_fused_sparse_forward(torch.empty(4, 1536), torch.arange(4), output, "missing")
    assert torch.count_nonzero(output) == 0


def test_fused_op_routes_to_layer_and_fake_has_no_side_effects():
    layer = SimpleNamespace(_run_fused_sparse_attention=Mock())
    context = SimpleNamespace(attn_metadata={}, no_compile_layers={"layer": layer})
    qkv, positions, output = torch.empty(4, 1536), torch.arange(4), torch.ones(4, 1024)
    with patch("vllm_ascend.models.minimax_m3.msa_m3.get_forward_context", return_value=context):
        minimax_m3_fused_sparse_forward(qkv, positions, output, "layer")
        minimax_m3_fused_sparse_forward_fake(qkv, positions, output, "layer")
    layer._run_fused_sparse_attention.assert_called_once_with(qkv, positions, output, None)
    assert torch.all(output == 1)


def test_fused_cache_insert_precedes_indexer_and_attention():
    layer = make_layer()
    layer.layer_name = "main"
    layer.num_heads, layer.num_kv_heads, layer.num_idx_heads = 8, 1, 1
    layer.kv_cache = torch.empty(2, 4, 16, 1, 128)
    index_cache = torch.empty(4, 16, 128)
    order = []
    layer.indexer = Mock(side_effect=lambda query: order.append("index") or "topk")
    layer.indexer.index_cache = SimpleNamespace(prefix="index", kv_cache=[index_cache])
    layer.impl = SimpleNamespace(forward=Mock(side_effect=lambda *args: order.append("attention")))
    layer._insert_kv = Mock(side_effect=AssertionError("duplicate cache insertion"))
    layer.rotary_emb.cos_sin_cache = torch.empty(32, 64)
    main_slots, index_slots = torch.tensor([3, -1, 8]), torch.tensor([7, 2, -1])
    context = SimpleNamespace(
        attn_metadata={
            "main": SimpleNamespace(slot_mapping=main_slots, num_actual_tokens=3),
            "index": SimpleNamespace(slot_mapping=index_slots),
        }
    )
    q, iq = torch.empty(4, 1024), torch.empty(4, 128)
    kernel = Mock(side_effect=lambda *args, **kwargs: order.append("insert") or (q, iq))
    module = SimpleNamespace(minimax_qknorm_rope_cache=kernel)
    with (
        patch.dict("sys.modules", {"vllm_ascend.ops.triton.linearnorm.minimax_qknorm_rope_cache": module}),
        patch("vllm_ascend.models.minimax_m3.minimax_m3.get_forward_context", return_value=context),
    ):
        MiniMaxM3SparseAttention._run_fused_sparse_attention(layer, torch.empty(4, 1536), torch.arange(4), q)
    assert order == ["insert", "index", "attention"]
    assert kernel.call_args.args[10] is main_slots
    assert kernel.call_args.args[11] is index_slots
    assert kernel.call_args.args[12] == 3
    layer._insert_kv.assert_not_called()
