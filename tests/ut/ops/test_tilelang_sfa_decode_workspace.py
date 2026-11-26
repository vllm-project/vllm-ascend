# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.tilelang.sfa_decode import runtime, weights
from vllm_ascend.ops.tilelang.sfa_decode.mtp import mtp_projection_rows, mtp_projection_tile


@pytest.mark.parametrize("requests", range(1, 61))
def test_mtp_projection_tiles_cover_target_verification_rows(requests):
    tokens = requests * 6
    rows = mtp_projection_rows(tokens)
    tile = mtp_projection_tile(rows)
    assert rows >= tokens and rows % tile == 0
    assert 32 <= tile <= 128
    if 128 <= rows <= 256:
        assert rows == 2 * tile


@pytest.mark.parametrize(
    "spec",
    [
        None,
        SimpleNamespace(method="mtp", num_speculative_tokens=1),
        SimpleNamespace(method="eagle3", num_speculative_tokens=5),
    ],
)
def test_other_decode_modes_use_native_attention(spec):
    impl = SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=spec))
    assert runtime.prepare_decode_layer(impl) is None


def test_mtp5_1m_bind_uses_worker_tables_and_graph_geometry(monkeypatch):
    meta = torch.device("meta")
    weights = SimpleNamespace(qkv_a=torch.empty(0, device=meta), index_heads=torch.empty(32, 4096, device=meta))
    monkeypatch.setattr(runtime, "prepare_decode_weights", lambda _: weights)
    manager = SimpleNamespace(get_persistent_resource=lambda _, allocate: allocate())
    monkeypatch.setattr(runtime, "current_workspace_manager", lambda: manager)
    impl = Mock()
    impl.indexer.impl.k_norm.eps = 1e-6
    impl.vllm_config.model_config.max_model_len = 1048576
    impl.vllm_config.scheduler_config.max_num_seqs = 60
    for tokens, padded_rows in ((216, 224), (324, 384), (360, 384)):
        impl.vllm_config.scheduler_config.max_num_batched_tokens = tokens
        layer = runtime.DecodeLayer(impl)
        assert layer.head_panel.shape == (28, 32, 176)
        assert layer.indices.shape == (padded_rows, 2176)
    compile_runner = Mock()
    monkeypatch.setattr(runtime, "MtpQueryAttention", compile_runner)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    pages = torch.empty((2, 4352), dtype=torch.uint8)
    keys = pages.view(torch.float8_e4m3fn).as_strided((2, 32, 1, 128), (4352, 128, 128, 1))
    scales = pages.view(torch.float32).as_strided((2, 32, 1, 1), (1088, 1, 1, 1), 1024)
    ring = torch.empty((2, 289, 256))[:, :9]
    ix = layer.indexer
    ix.topk_indices_buffer = None
    ix.tail_cache.prefix = "tail"
    ix._bound_caches.side_effect = lambda cache: (keys, scales) if cache is ix.k_cache else (ring,)
    tail = SimpleNamespace()
    context = SimpleNamespace(attn_metadata={"tail": tail})
    monkeypatch.setattr(runtime, "get_forward_context", lambda: context)
    main_cache = torch.empty((2, 128, 1, 528), dtype=torch.float8_e4m3fn, device=meta)

    def integers(shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device=meta)

    graph_shapes = [(r, 6 * r, 8228, 8228) for r in range(1, 61)]
    graph_shapes.extend([(8, 11, 8192, 8256), (60, 64, 8192, 8256), (60, 360, 8192, 8256)])
    for requests, tokens, main_cols, index_cols in graph_shapes:
        tail.block_table = integers((requests, 32))
        tail.cann_inputs = SimpleNamespace(write_indices=integers((requests, 6), torch.int64))
        md = SimpleNamespace(
            num_prefills=0,
            max_query_len=6,
            block_table=integers((requests, main_cols)),
            positions=integers((tokens,), torch.int64),
            query_start_loc=integers((requests + 1,)),
        )
        idx = SimpleNamespace(
            write_metadata=None,
            indexer_row_shard=None,
            block_table=integers((requests, index_cols)),
            raw_seq_lens=integers((requests,)),
            cum_query_lens=integers((requests,), torch.int64),
            slot_mapping=integers((tokens,), torch.int64),
        )
        hidden = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=meta)
        args = hidden, torch.empty_like(hidden), main_cache, integers((tokens,)), md, idx
        runner, _, ids = layer.bind(*args)
        assert runner is compile_runner.return_value and ids.shape == (tokens, 2176)
        assert compile_runner.call_args.kwargs["tokens"] == tokens
        assert compile_runner.call_args.kwargs["table_columns"] == (main_cols, index_cols)
        assert compile_runner.call_args.kwargs["model_context"] == 1048576
        with monkeypatch.context() as capture:
            capture.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
            assert layer.bind(*args)[0] is runner
        md.block_table = integers((requests, 8191))
        assert layer.bind(*args) is None
        md.block_table = integers((requests, main_cols))
        idx.block_table = integers((requests, 8191))
        assert layer.bind(*args) is None
    assert compile_runner.call_count == len(graph_shapes)


@pytest.mark.parametrize(
    ("native_enabled", "weight_format", "padding", "aliases_weight"),
    [(True, 29, 0, True), (False, 29, 0, False), (True, 2, 0, False), (True, 29, 32, False)],
)
def test_mtp_output_weight_reuse(monkeypatch, native_enabled, weight_format, padding, aliases_weight):
    inputs, outputs = 64, 32
    storage = torch.arange(inputs * outputs + padding, dtype=torch.int32).to(torch.uint8)
    weight = storage[: inputs * outputs].view(inputs, outputs).view(torch.float8_e4m3fn)
    scales = torch.zeros((inputs // 64, outputs, 2), dtype=torch.uint8)
    layer = SimpleNamespace(weight=weight, weight_scale=scales)
    monkeypatch.setattr(
        weights,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda t: weight_format if t.dtype == weight.dtype else 2,
            npu_format_cast=lambda t, fmt: t,
        ),
    )

    codes, packed_scales = weights._mx_weight(layer, outputs, inputs, allow_native_output=native_enabled)

    assert (codes.data_ptr() == weight.data_ptr()) == aliases_weight
    expected = weight if aliases_weight else weight.T.contiguous()
    assert torch.equal(codes.view(torch.uint8), expected.view(torch.uint8))
    assert packed_scales.data_ptr() == scales.data_ptr()


@pytest.mark.parametrize("context", [131072, 1048576])
def test_mtp_native_projection_layout_matches_context(monkeypatch, context):
    def matrix(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, device="meta", dtype=dtype)

    def projection(inputs, outputs):
        return SimpleNamespace(
            weight=matrix(inputs, outputs, dtype=torch.float8_e4m3fn),
            weight_scale=matrix(inputs // 64, outputs, 2, dtype=torch.uint8),
        )

    norm = SimpleNamespace(weight=matrix(128), bias=matrix(128))
    indexer = SimpleNamespace(
        wk_weights_proj=SimpleNamespace(weight=matrix(160, 4096)),
        head_dim=128,
        index_kpool_compress_gate=matrix(128, 4096),
        k_norm=norm,
        index_kpool_compress_ape=matrix(4, 128),
        wq_b=SimpleNamespace(weight=matrix(4096, 1536)),
    )
    impl = SimpleNamespace(
        local_num_heads=64,
        qk_rope_head_dim=0,
        runtime_has_indexer=True,
        g_proj=None,
        indexer=SimpleNamespace(impl=indexer),
        fused_qkv_a_proj=projection(4096, 2048),
        q_proj=projection(1536, 16384),
        o_proj=projection(16384, 4096),
        W_UK_T=matrix(64, 256, 512),
        W_UV=matrix(64, 512, 256),
        q_a_layernorm=norm,
        kv_a_layernorm=norm,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=context)),
    )
    monkeypatch.setattr(
        weights,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda t: 29 if t.dtype == torch.float8_e4m3fn else 2,
            npu_format_cast=lambda t, fmt: t,
        ),
    )

    result = weights.prepare_decode_weights(impl)

    if context == 1048576:
        assert result.qkv_a.shape == impl.fused_qkv_a_proj.weight.shape
        assert result.q_b.shape == impl.q_proj.weight.shape
        assert result.out.shape == impl.o_proj.weight.shape
        assert result.w_uk.shape == impl.W_UK_T.shape and result.w_uv.shape == impl.W_UV.shape
    else:
        assert result.qkv_a.shape == impl.fused_qkv_a_proj.weight.T.shape
        assert result.q_b.shape == impl.q_proj.weight.T.shape
        assert result.out.shape == impl.o_proj.weight.T.shape
        assert result.w_uk.shape == impl.W_UK_T.mT.shape and result.w_uv.shape == impl.W_UV.mT.shape
