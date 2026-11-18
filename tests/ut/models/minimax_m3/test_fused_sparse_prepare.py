# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3SparseAttention
from vllm_ascend.models.minimax_m3.msa_m3 import (
    minimax_m3_fused_sparse_forward,
    minimax_m3_fused_sparse_forward_fake,
)
from vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache import (
    _minimax_qknorm_rope_cache_kernel,
    minimax_qknorm_rope_cache,
)


def make_layer():
    norm = SimpleNamespace(variance_epsilon=1e-6, weight=torch.zeros(128, dtype=torch.bfloat16))
    return SimpleNamespace(
        use_dspark=False,
        indexer=SimpleNamespace(impl=SimpleNamespace(use_mxfp4=False)),
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
    "invalid",
    [
        "key_dtype",
        "key_shape",
        "key_stride",
        "dual_cache",
        "negative_actual",
        "excess_actual",
        "main_slots",
        "index_slots",
        "positions",
        "norm_dtype",
        "norm_stride",
    ],
)
def test_materialized_prepare_rejects_invalid_metadata_before_device_launch(invalid):
    tokens, dim = 4, 128
    weights = [torch.zeros(dim, dtype=torch.bfloat16) for _ in range(4)]
    args = dict(
        packed=torch.zeros((tokens, 12 * dim), dtype=torch.bfloat16),
        cos_sin_cache=torch.zeros((tokens, 64), dtype=torch.bfloat16),
        positions=torch.arange(tokens),
        key_cache=torch.zeros((1, 16, 1, dim), dtype=torch.bfloat16),
        value_cache=torch.zeros((1, 16, 1, dim), dtype=torch.bfloat16),
        index_cache=None,
        slot_mapping=torch.arange(tokens),
        index_slot_mapping=torch.arange(tokens),
        num_actual_tokens=tokens,
        num_q_heads=8,
        num_kv_heads=1,
        num_index_heads=1,
        eps=1e-6,
        index_key_out=torch.empty((tokens, dim), dtype=torch.bfloat16),
    )
    if invalid == "key_dtype":
        args["index_key_out"] = args["index_key_out"].float()
    elif invalid == "key_shape":
        args["index_key_out"] = args["index_key_out"][:1]
    elif invalid == "key_stride":
        args["index_key_out"] = torch.empty((tokens, dim * 2), dtype=torch.bfloat16)[:, ::2]
    elif invalid == "dual_cache":
        args["index_cache"] = torch.empty((1, 16, dim), dtype=torch.bfloat16)
    elif invalid == "negative_actual":
        args["num_actual_tokens"] = -1
    elif invalid == "excess_actual":
        args["num_actual_tokens"] = tokens + 1
    elif invalid in ("main_slots", "index_slots"):
        args["slot_mapping" if invalid == "main_slots" else "index_slot_mapping"] = torch.arange(1)
    elif invalid == "positions":
        args["positions"] = torch.arange(1)
    elif invalid == "norm_dtype":
        weights[2] = weights[2].float()
    elif invalid == "norm_stride":
        weights[2] = torch.zeros(dim * 2, dtype=torch.bfloat16)[::2]
    args.update(zip(("q_weight", "k_weight", "index_q_weight", "index_k_weight"), weights))
    with patch("vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache.init_device_properties_triton") as initialize:
        with pytest.raises(AssertionError):
            minimax_qknorm_rope_cache(**args)
        initialize.assert_not_called()


def test_empty_materialized_prepare_never_initializes_device():
    dim = 128
    weight = torch.zeros(dim, dtype=torch.bfloat16)
    with patch("vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache.init_device_properties_triton") as initialize:
        query, index_query = minimax_qknorm_rope_cache(
            torch.empty((0, 12 * dim), dtype=torch.bfloat16),
            torch.empty((1, 64), dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.long),
            weight,
            weight,
            weight,
            weight,
            torch.empty((1, 16, 1, dim), dtype=torch.bfloat16),
            torch.empty((1, 16, 1, dim), dtype=torch.bfloat16),
            None,
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            0,
            8,
            1,
            1,
            1e-6,
            index_key_out=torch.empty((0, dim), dtype=torch.bfloat16),
        )
        assert query.shape == (0, 8 * dim) and index_query.shape == (0, dim)
        initialize.assert_not_called()


def test_materialized_cache_layout_constants_do_not_follow_request_size():
    dim = 128
    weight = torch.zeros(dim, dtype=torch.bfloat16)
    layouts = []
    kernel_fn = getattr(_minimax_qknorm_rope_cache_kernel, "fn", _minimax_qknorm_rope_cache_kernel)
    parameters = inspect.signature(kernel_fn).parameters
    for tokens in (1, 224, 1664):
        with (
            patch("vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache.init_device_properties_triton"),
            patch("vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache.get_vectorcore_num", return_value=64),
            patch("vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache._minimax_qknorm_rope_cache_kernel") as kernel,
        ):
            minimax_qknorm_rope_cache(
                torch.empty((tokens, 12 * dim), dtype=torch.bfloat16),
                torch.empty((2048, 64), dtype=torch.bfloat16),
                torch.arange(tokens),
                weight,
                weight,
                weight,
                weight,
                torch.empty((16, 128, 1, dim), dtype=torch.bfloat16),
                torch.empty((16, 128, 1, dim), dtype=torch.bfloat16),
                None,
                torch.arange(tokens),
                torch.arange(tokens),
                tokens,
                8,
                1,
                1,
                1e-6,
                index_key_out=torch.empty((tokens, dim), dtype=torch.bfloat16),
            )
            arguments = dict(zip(parameters, kernel.__getitem__.return_value.call_args.args))
            # Strides and block width passed to the JIT for the temporary key.
            layouts.append((arguments["I_BLOCK"], arguments["I_TOKEN"], arguments["INDEX_BLOCK_SIZE"]))
            assert arguments["NUM_TOKENS"] == tokens
    assert layouts == [(dim, dim, 1)] * 3


@pytest.mark.parametrize(
    "unsupported",
    [None, "dtype", "cpu", "positions", "heads", "rope", "eps", "triton", "large_batch", "cache_dtype"],
)
@pytest.mark.parametrize("use_dspark", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_mxfp4", [False, True])
def test_fusion_guard(unsupported, use_dspark, cache_dtype, use_mxfp4):
    layer = make_layer()
    layer.use_dspark = use_dspark
    layer.indexer.impl.use_mxfp4 = use_mxfp4
    layer.kv_cache_torch_dtype = cache_dtype
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
        layer.k_norm = SimpleNamespace(variance_epsilon=1e-5, weight=torch.zeros(128, dtype=torch.bfloat16))
    elif unsupported == "large_batch":
        hidden.shape = (262145, 128)
    elif unsupported == "cache_dtype":
        layer.kv_cache_torch_dtype = torch.float8_e5m2
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


def test_mxfp4_fusion_falls_back_for_float32_norm_weights():
    layer = make_layer()
    layer.indexer.impl.use_mxfp4 = True
    layer.index_k_norm = SimpleNamespace(variance_epsilon=1e-6, weight=torch.zeros(128))
    hidden = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, shape=(4, 128))
    with patch("vllm_ascend.models.minimax_m3.minimax_m3.HAS_TRITON", True):
        assert not MiniMaxM3SparseAttention._can_fuse_sparse_prepare(layer, torch.arange(4), hidden)


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


@pytest.mark.parametrize("use_dspark", [False, True])
@pytest.mark.parametrize("use_mxfp4", [False, True])
def test_fused_cache_insert_precedes_indexer_and_attention(use_dspark, use_mxfp4):
    layer = make_layer()
    layer.use_dspark = use_dspark
    layer.layer_name = "main"
    layer.num_heads, layer.num_kv_heads, layer.num_idx_heads = 8, 1, 1
    layer.kv_cache = torch.empty(2, 4, 16, 1, 128)
    index_cache = torch.empty(4, 16, 128)
    order = []

    packed_query = (torch.empty(3, 1, 64, dtype=torch.uint8), torch.empty(3, 1, 4, dtype=torch.uint8))

    def run_indexer(query, prequantized=None):
        order.append("index")
        assert prequantized is packed_query if use_mxfp4 else prequantized is None
        return "topk"

    layer.indexer = Mock(side_effect=run_indexer)
    layer.indexer.impl = SimpleNamespace(use_mxfp4=use_mxfp4)
    layer.indexer.index_cache = SimpleNamespace(prefix="index", kv_cache=[index_cache])
    layer.impl = SimpleNamespace(forward=Mock(side_effect=lambda *args: order.append("attention")))
    layer._insert_kv = Mock(side_effect=AssertionError("duplicate cache insertion"))
    layer.rotary_emb.cos_sin_cache = torch.empty(32, 64)
    main_slots, index_slots = torch.tensor([3, -1, 8]), torch.tensor([7, 2, -1])
    context = SimpleNamespace(
        attn_metadata={
            "main": SimpleNamespace(slot_mapping=main_slots, num_actual_tokens=3),
            "index": SimpleNamespace(slot_mapping=index_slots, num_actual_tokens=3),
        }
    )
    q, iq = torch.empty(4, 1024), torch.empty(4, 128)

    def run_kernel(*args, **kwargs):
        order.append("insert")
        return q, iq

    kernel = Mock(side_effect=run_kernel)
    module = SimpleNamespace(minimax_qknorm_rope_cache=kernel)

    def insert_mxfp4(cache, qk, heads, slots):
        order.append("quantize_insert")
        assert cache is layer.indexer.index_cache.kv_cache
        assert qk.shape == (8, 128) and heads == 1 and torch.equal(slots, index_slots)
        return packed_query

    with (
        patch.dict("sys.modules", {"vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache": module}),
        patch("vllm_ascend.models.minimax_m3.minimax_m3.get_forward_context", return_value=context),
        patch(
            "vllm_ascend.models.minimax_m3.ops.msa_m3_mxfp4.pack_mxfp4_query_and_insert_cache",
            side_effect=insert_mxfp4,
        ),
    ):
        MiniMaxM3SparseAttention._run_fused_sparse_attention(layer, torch.empty(4, 1536), torch.arange(4), q)
    assert order == (
        ["insert", "quantize_insert", "index", "attention"] if use_mxfp4 else ["insert", "index", "attention"]
    )
    assert kernel.call_args.args[10] is main_slots
    assert kernel.call_args.args[11] is index_slots
    assert kernel.call_args.args[12] == 3
    if use_mxfp4:
        assert kernel.call_args.args[9] is None
        assert kernel.call_args.kwargs["index_key_out"].shape == (4, 128)
        query_out = kernel.call_args.kwargs["index_query_out"]
        key_out = kernel.call_args.kwargs["index_key_out"]
        assert query_out.shape == (4, 128)
        assert query_out.data_ptr() + query_out.numel() * query_out.element_size() == key_out.data_ptr()
    else:
        assert kernel.call_args.args[9] is index_cache
        assert kernel.call_args.kwargs["index_key_out"] is None
    layer._insert_kv.assert_not_called()
