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
    "unsupported",
    [None, "dtype", "cpu", "positions", "heads", "rope", "eps", "triton", "large_batch", "cache_dtype"],
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
        hidden.shape = (262145, 128)
    elif unsupported == "cache_dtype":
        layer.kv_cache_torch_dtype = torch.float8_e5m2
    with patch("vllm_ascend.models.minimax_m3.minimax_m3.HAS_TRITON", unsupported != "triton"):
        assert MiniMaxM3SparseAttention._can_fuse_sparse_prepare(layer, positions, hidden) == (unsupported is None)


@pytest.mark.parametrize("separate_index", [False, True])
def test_fused_forward_passes_projection_views(separate_index):
    hidden = torch.randn(4, 128, dtype=torch.bfloat16)
    qkv = torch.randn(4, 1280 if separate_index else 1536, dtype=torch.bfloat16)
    index_qk = torch.randn(4, 256, dtype=torch.bfloat16) if separate_index else None
    positions = torch.arange(4)
    layer = make_layer()
    layer._can_fuse_sparse_prepare = Mock(return_value=True)
    layer.qkv_proj = Mock(return_value=(qkv, None))
    layer.indexer_proj = Mock(return_value=(index_qk, None)) if separate_index else None
    layer._sparse_prepare = Mock(side_effect=AssertionError("separate prepare must not run"))
    layer.q_size, layer.kv_size, layer.index_q_size = 1024, 128, 128
    layer.layer_name = "layers.0.attn"
    layer.o_proj = Mock(side_effect=lambda x: (x, None))

    with patch.object(torch.ops.vllm, "minimax_m3_fused_sparse_forward") as fused:
        output = MiniMaxM3SparseAttention.forward(layer, positions, hidden)

    layer.qkv_proj.assert_called_once_with(hidden)
    layer._sparse_prepare.assert_not_called()
    if separate_index:
        layer.indexer_proj.assert_called_once_with(hidden)
        expected = (qkv[:, :1024], qkv[:, 1024:1152], qkv[:, 1152:], index_qk[:, :128], index_qk[:, 128:])
    else:
        expected = (qkv[:, :1024], qkv[:, 1024:1152], qkv[:, 1152:1280], qkv[:, 1280:1408], qkv[:, 1408:])

    fused.assert_called_once()
    args = fused.call_args.args
    for actual, view in zip(args[:5], expected, strict=True):
        assert actual.data_ptr() == view.data_ptr()
        assert actual.shape == view.shape
        assert actual.stride() == view.stride()
    assert args[5] is positions
    assert args[6] is output
    assert args[7] == layer.layer_name
    assert output.shape == (4, 1024) and output.dtype == hidden.dtype


@pytest.mark.parametrize(
    "tokens, supported",
    [
        (512, True),
        (513, True),
        (8192, True),
        (32768, True),
        (65536, True),
        (131072, True),
        (262144, True),
        (262145, False),
    ],
)
def test_fusion_token_limit(tokens, supported):
    hidden = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, shape=(tokens, 128))
    with patch("vllm_ascend.models.minimax_m3.minimax_m3.HAS_TRITON", True):
        assert (
            MiniMaxM3SparseAttention._can_fuse_sparse_prepare(make_layer(), torch.arange(tokens), hidden) == supported
        )


def test_fused_op_dummy_does_not_touch_cache():
    projections = tuple(torch.empty(4, width) for width in (1024, 128, 128, 128, 128))
    output = torch.ones(4, 1024)
    context = SimpleNamespace(attn_metadata=None, no_compile_layers={})
    with patch("vllm_ascend.models.minimax_m3.msa_m3.get_forward_context", return_value=context):
        minimax_m3_fused_sparse_forward(*projections, torch.arange(4), output, "missing")
    assert torch.count_nonzero(output) == 0


def test_fused_op_routes_to_layer_and_fake_has_no_side_effects():
    layer = SimpleNamespace(_run_fused_sparse_attention=Mock())
    context = SimpleNamespace(attn_metadata={}, no_compile_layers={"layer": layer})
    projections = tuple(torch.empty(4, width) for width in (1024, 128, 128, 128, 128))
    positions, output = torch.arange(4), torch.ones(4, 1024)
    with patch("vllm_ascend.models.minimax_m3.msa_m3.get_forward_context", return_value=context):
        minimax_m3_fused_sparse_forward(*projections, positions, output, "layer")
        minimax_m3_fused_sparse_forward_fake(*projections, positions, output, "layer")
    layer._run_fused_sparse_attention.assert_called_once_with(*projections, positions, output)
    assert torch.all(output == 1)


def test_fused_cache_insert_precedes_indexer_and_attention():
    layer = make_layer()
    layer.layer_name = "main"
    layer.kv_cache = torch.empty(2, 4, 16, 1, 128)
    index_cache = torch.empty(4, 16, 128)
    order = []

    def run_indexer(query):
        order.append("index")
        return "topk"

    layer.indexer = Mock(side_effect=run_indexer)
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

    def run_kernel(*args, **kwargs):
        order.append("insert")
        return q, iq

    kernel = Mock(side_effect=run_kernel)
    projections = torch.empty(4, 1536).split((1024, 128, 128, 128, 128), dim=-1)
    module = SimpleNamespace(minimax_qknorm_rope_cache=kernel)
    with (
        patch.dict("sys.modules", {"vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache": module}),
        patch("vllm_ascend.models.minimax_m3.minimax_m3.get_forward_context", return_value=context),
    ):
        MiniMaxM3SparseAttention._run_fused_sparse_attention(layer, *projections, torch.arange(4), q)
    assert order == ["insert", "index", "attention"]
    args = kernel.call_args.args
    for actual, view in zip(args[:5], projections, strict=True):
        assert actual is view
    assert args[14] is main_slots
    assert args[15] is index_slots
    assert args[16] == 3
    layer._insert_kv.assert_not_called()
